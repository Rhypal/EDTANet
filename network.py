import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from tqdm import tqdm
from clip.model import ModifiedResNet, VisionTransformer


# === 梯度反转层 (用于域对抗) ===
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# === 空间注意力桥 ===
class SpatialAttentionBridge(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, L, D] -> [B, D, H, W]
        B, L, D = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).view(B, D, H, W)
        return self.conv(x)  # [B, 1, H, W]


# === 主模型 ===
class DAModel(nn.Module):
    def __init__(self, num_classes, clip_weight_path, device='cuda:0'):
        super().__init__()
        self.device = device
        self.feat_dim = 512     # Vit-16  Vit-32=768
        self.num_classes = num_classes

        # 1. 加载 CLIP (冻结)
        print(f"Loading CLIP from {clip_weight_path}...")
        self.clip_model, _ = clip.load(clip_weight_path, device=device, jit=False)
        self.clip_model.float()
        for p in self.clip_model.parameters(): p.requires_grad = False
        # 部分解冻
        for p in self.clip_model.visual.ln_post.parameters(): p.requires_grad = True
        for p in self.clip_model.visual.transformer.resblocks[-1].parameters(): p.requires_grad = True

        # 2. 解耦适配器 (Decoupling Adapter)
        # 将 Patch 特征解耦为 Domain-Invariant (inv) 和 Domain-Specific (spec)
        self.adapter = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim * 2),
            # nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 3. 域判别器 (作用于 V_spec)
        self.discriminator = nn.Sequential(
            nn.Linear(self.feat_dim, 256), nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

        # 4. 跨模态对齐 (Cross-Space Alignment)
        self.cross_attn = nn.MultiheadAttention(self.feat_dim, num_heads=8, batch_first=True)
        self.spatial_bridge = SpatialAttentionBridge(self.feat_dim)

        # 5. 原型分类器 (Cosine Classifier)
        # 权重 W 即为 learnable prototypes
        self.prototypes = nn.Parameter(torch.randn(num_classes, self.feat_dim))
        # nn.init.xavier_uniform_(self.prototypes)
        nn.init.normal_(self.adapter[0].weight, std=0.01)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    # === 原型初始化函数 ===
    def init_prototypes_with_source(self, source_loader, text_embeddings, device):
        """用源域数据的特征中心初始化原型"""
        print(">>> Warmup: Initializing prototypes with source domain features...")
        self.eval()

        # 临时存储每个类别的特征
        all_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels, _ in tqdm(source_loader, desc="Extracting Source Features"):
                imgs = imgs.to(device)
                # 复用前向传播拿到 f_weighted (加权后的视觉特征)
                out = self.forward(imgs, text_embeddings)
                feats = out['f_weighted']  # [B, 512]

                # 归一化特征 (因为原型也是归一化的)
                feats = F.normalize(feats, dim=1)

                all_feats.append(feats.cpu())
                all_labels.append(labels.cpu())

        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 计算每个类别的中心
        new_prototypes = torch.zeros_like(self.prototypes)
        for i in range(self.num_classes):
            # 找到属于类别 i 的所有样本索引
            idxs = (all_labels == i).nonzero(as_tuple=True)[0]
            if len(idxs) > 0:
                class_feats = all_feats[idxs]
                # 计算均值并归一化
                mean_feat = class_feats.mean(dim=0)
                mean_feat = F.normalize(mean_feat, dim=0)
                new_prototypes.data[i] = mean_feat.to(device)
            else:
                print(f"Warning: Class {i} has no samples in source domain!")

        print(">>> Prototypes initialized successfully!")

    def compute_uncertainty(self, logits):
        """
        基于 EDL 计算不确定性
        u = K / sum(alpha), where alpha = exp(logits) + 1
        """
        # evidence = torch.exp(logits)
        logits = logits / 0.5
        logits_center = logits - logits.mean(dim=1, keepdim=True)
        evidence = F.softplus(logits_center)
        evidence = torch.clamp(evidence, min=1e-5, max=1e3)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = self.num_classes / S
        return uncertainty, alpha

    def forward(self, img, text_embeddings):
        """
        img: [B, 3, 224, 224]   img torch.Size([16, 3, 224, 224])
        text_embeddings: [Num_Classes, 512] (聚合后的增强文本特征)    text ViT-16=torch.Size([27, 512])
        """
        # A. 图像编码 (获取 Patch 特征)
        with torch.no_grad():
            # 获取 CLIP 最后一层 Transformer 输出 [B, L+1, D]
            x = self.clip_model.visual.conv1(img.type(self.clip_model.dtype))  # [B, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B,grid ** 2,width]  (16,196,768)
            x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype)
                           + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
                          dim=1)  # [B, L+1, D]     (16,197,768)
            x = x + self.clip_model.visual.positional_embedding.to(x.dtype)

            x = self.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            visual_patches = self.clip_model.visual.ln_post(x)[:, 1:, :].float()  # [B, 49, 512] (去除CLS)
            cls_token = self.clip_model.visual.ln_post(x[:, 0, :]).float()
        if self.clip_model.visual.proj is not None:
            visual_patches = visual_patches@self.clip_model.visual.proj
            cls_token = cls_token @ self.clip_model.visual.proj  # [B, 512]
        B, L, D = visual_patches.shape      # torch.Size([16, 196, 512])

        # B. 特征解耦
        features = self.adapter(visual_patches)  # [B, L, 1024]
        delta_inv, delta_spec = torch.split(features, self.feat_dim, dim=2)  # [B, L, 512]
        v_inv = visual_patches + 0.2 * delta_inv
        # v_inv, v_spec = torch.split(features, self.feat_dim, dim=2)  # [B, L, 512]
        v_spec = delta_spec

        # C. 域对抗 (Domain Adversarial)
        v_spec_pool = v_spec.mean(dim=1)  # [B, 512]
        # disc_pred = self.discriminator(grad_reverse(v_inv_pool, alpha))
        disc_pred = self.discriminator(v_spec_pool)

        # D. 拓扑逻辑与跨模态注意力
        # Text Embeddings 作为 Key/Value, V_inv 作为 Query
        # 扩展 text 以匹配 batch: [B, Num_Classes, 512]
        k_v_text = text_embeddings.unsqueeze(0).expand(B, -1, -1)

        # Cross Attention: 图像 Patch 寻找对应的 文本描述
        # csa_out: [B, L, 512], attn_weights: [B, L, Num_Classes]
        csa_out, attn_weights = self.cross_attn(v_inv, k_v_text, k_v_text)

        # === 基于不确定性的动态加权 ===
        # 1. 先计算“纯视觉”特征的不确定性, 使用 v_inv 的均值与原型进行一次粗糙分类
        v_inv_mean = v_inv.mean(dim=1)
        f_norm_vis = F.normalize(v_inv_mean, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_vis = logit_scale * (f_norm_vis @ p_norm.t())

        # 计算视觉不确定性 (Uncertainty of Visual Features)
        u_vis, _ = self.compute_uncertainty(logits_vis)  # [B, 1]

        # 2. 动态调整权重
        # 如果不确定性 u_vis 大（图像质量差），则增大文本特征 csa_out 的权重, 基础权重设为 1.0，根据不确定性进行缩放
        w_text = 1.0 + 2.0 * u_vis.view(B, 1, 1)

        # 融合特征: v_inv (视觉) + w_text * csa_out (文本引导)
        f_aligned = v_inv + w_text * csa_out
        # f_aligned = v_inv + csa_out

        # 生成拓扑损失所需的两个 Map
        # 1. m_vis (视觉热力图): 网络自己学出来的重点区域
        m_vis = self.spatial_bridge(f_aligned)  # [B, 1, 7, 7]

        # 2. m_txt (文本引导图): 文本认为哪些 Patch 重要
        # 简化为 Max Response，代表“任意病害描述”激活的区域
        m_txt_raw = attn_weights.max(dim=2)[0]  # [B, 49]
        H_w = int(L ** 0.5)
        m_txt = m_txt_raw.view(B, 1, H_w, H_w).detach()  # [B, 1, 7, 7] (Detach作为GT)

        # E. 分类 (使用原型)
        # 视觉注意力加权池化
        # f_final: [B, 512]
        f_weighted = (f_aligned * m_vis.view(B, L, 1)).sum(dim=1)
        # 既保留了空间解耦设计，又利用了 CLIP 强大的全局对齐能力
        f_final = f_weighted + cls_token
        # 归一化用于余弦相似度
        f_norm = F.normalize(f_final, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)

        # Cosine Similarity Logits (scaled)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * (f_norm @ p_norm.t())
        # 计算最终的不确定性 (用于伪标签筛选)
        uncertainty_final, alpha_dist = self.compute_uncertainty(logits)

        return {
            "logits": logits,  # 分类结果
            "uncertainty": uncertainty_final,  # [B, 1]
            "alpha": alpha_dist,  # [B, Num_Classes] 用于 loss
            "visual_uncertainty": u_vis,  # 记录一下视觉不确定性，方便调试
            "attn_weights": attn_weights,
            "disc_pred": disc_pred,  # 域判别结果
            "origin_visual": visual_patches,
            "v_inv": v_inv,  # 用于解耦损失
            "v_spec": v_spec,  # 用于解耦损失
            "m_vis": m_vis,  # 用于拓扑损失
            "m_txt": m_txt,  # 用于拓扑损失
            "f_weighted": f_weighted,  # 用于原型损失 (特征)
            "prototypes": self.prototypes  # 用于原型损失 (中心)
        }


class DAModel_backbone(nn.Module):
    def __init__(self, num_classes, clip_model_name, clip_weight_path, device='cuda:0'):
        super().__init__()
        self.device = device
        self.feat_dim = 512     # 文本维度被 ViT-B/16 锁定为 512
        self.num_classes = num_classes

        # 1. 加载 CLIP (冻结)
        print(f"Loading CLIP from {clip_model_name}...")
        # self.clip_model, _ = clip.load(clip_weight_path, device=device, jit=False)
        self.clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
        self.clip_model.float()

        # 判断 Backbone 类型
        if isinstance(self.clip_model.visual, ModifiedResNet):
            self.backbone_type = 'ResNet'
            self.visual_raw_dim = self._detect_resnet_dim()  #  RN50=2048
            print("-> Detected Architecture: ResNet (RN)")
        elif isinstance(self.clip_model.visual, VisionTransformer):
            self.backbone_type = 'ViT'
            self.visual_raw_dim = self.clip_model.visual.width  #  ViT-L=1024, ViT-B=768
            print("-> Detected Architecture: Vision Transformer (ViT)")
        else:
            raise ValueError("Unknown CLIP Backbone Architecture")

        if self.visual_raw_dim != self.feat_dim:
            print(f"-> Creating Projection: {self.visual_raw_dim} -> {self.feat_dim}")
            self.visual_proj = nn.Linear(self.visual_raw_dim, self.feat_dim)
            nn.init.kaiming_normal_(self.visual_proj.weight)  # 必须重新初始化，因为这是一层新层
            nn.init.zeros_(self.visual_proj.bias)
        else:
            if clip_model_name == 'ViT-B/16':
                self.visual_proj = nn.Identity()
            else:
                self.visual_proj = nn.Linear(self.visual_raw_dim, self.feat_dim)

        for p in self.clip_model.parameters(): p.requires_grad = False
        # 部分解冻策略 (根据需要调整)
        if self.backbone_type == 'ViT':
            for p in self.clip_model.visual.ln_post.parameters(): p.requires_grad = True
            for p in self.clip_model.visual.transformer.resblocks[-1].parameters(): p.requires_grad = True
        else:  # ResNet
            for p in self.clip_model.visual.layer4.parameters(): p.requires_grad = True
            if hasattr(self.clip_model.visual, 'attnpool'):  # RN 的 attnpool 参数
                for p in self.clip_model.visual.attnpool.parameters(): p.requires_grad = True

        # 2. 解耦适配器 (Decoupling Adapter)
        # 将 Patch 特征解耦为 Domain-Invariant (inv) 和 Domain-Specific (spec)
        self.adapter = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim * 2), # nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 3. 域判别器 (作用于 V_spec)
        self.discriminator = nn.Sequential(
            nn.Linear(self.feat_dim, 256), nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

        # 4. 跨模态对齐 (Cross-Space Alignment)
        self.cross_attn = nn.MultiheadAttention(self.feat_dim, num_heads=8, batch_first=True)
        self.spatial_bridge = SpatialAttentionBridge(self.feat_dim)

        # 5. 原型分类器 (Cosine Classifier)
        # 权重 W 即为 learnable prototypes
        self.prototypes = nn.Parameter(torch.randn(num_classes, self.feat_dim))
        # nn.init.xavier_uniform_(self.prototypes)
        nn.init.normal_(self.adapter[0].weight, std=0.01)
        nn.init.zeros_(self.adapter[0].bias)

    def _detect_resnet_dim(self):
        """探测 ResNet layer4 的通道数 (适配 CLIP ModifiedResNet 的 Stem 结构)"""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            x = dummy.type(self.clip_model.dtype)
            x = self.clip_model.visual.relu1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
            x = self.clip_model.visual.relu2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
            x = self.clip_model.visual.relu3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
            x = self.clip_model.visual.avgpool(x)

            x = self.clip_model.visual.layer1(x)
            x = self.clip_model.visual.layer2(x)
            x = self.clip_model.visual.layer3(x)
            x = self.clip_model.visual.layer4(x)
            return x.shape[1]  # 返回 Channels

    def _get_visual_features(self, img):
        """提取特征并映射到 512 维 (适配 CLIP ModifiedResNet 的 Stem 结构)"""
        if self.backbone_type == 'ResNet':
            # ResNet 前向
            x = img.type(self.clip_model.dtype)
            x = self.clip_model.visual.relu1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
            x = self.clip_model.visual.relu2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
            x = self.clip_model.visual.relu3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
            x = self.clip_model.visual.avgpool(x)
            x = self.clip_model.visual.layer1(x)
            x = self.clip_model.visual.layer2(x)
            x = self.clip_model.visual.layer3(x)
            feat_map = self.clip_model.visual.layer4(x)  # [B, C_raw, H, W]
            # 展平: [B, C_raw, H*W] -> [B, H*W, C_raw]
            B, C, H, W = feat_map.shape
            visual_patches = feat_map.reshape(B, C, -1).permute(0, 2, 1)  # [B, L, C_raw]

        else:  # ViT (保持不变)
            x = self.clip_model.visual.conv1(img.type(self.clip_model.dtype))
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype)
                           + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
            x = self.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.clip_model.visual.transformer(x)
            x = x.permute(1, 0, 2)
            visual_patches = self.clip_model.visual.ln_post(x)[:, 1:, :]  # 映射到 512
        visual_patches = visual_patches.float()
        visual_patches = self.visual_proj(visual_patches)

        return visual_patches

    # === 原型初始化函数 ===
    def init_prototypes_with_source(self, source_loader, text_embeddings, device):
        """用源域数据的特征中心初始化原型"""
        print(">>> Warmup: Initializing prototypes with source domain features...")
        self.eval()

        # 临时存储每个类别的特征
        all_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels, _ in tqdm(source_loader, desc="Extracting Source Features"):
                imgs = imgs.to(device)
                # 复用前向传播拿到 f_weighted (加权后的视觉特征)
                out = self.forward(imgs, text_embeddings)
                feats = out['f_weighted']  # [B, 512]

                # 归一化特征 (因为原型也是归一化的)
                feats = F.normalize(feats, dim=1)

                all_feats.append(feats.cpu())
                all_labels.append(labels.cpu())

        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 计算每个类别的中心
        new_prototypes = torch.zeros_like(self.prototypes)
        for i in range(self.num_classes):
            # 找到属于类别 i 的所有样本索引
            idxs = (all_labels == i).nonzero(as_tuple=True)[0]
            if len(idxs) > 0:
                class_feats = all_feats[idxs]
                # 计算均值并归一化
                mean_feat = class_feats.mean(dim=0)
                mean_feat = F.normalize(mean_feat, dim=0)
                new_prototypes.data[i] = mean_feat.to(device)
            else:
                print(f"Warning: Class {i} has no samples in source domain!")

        print(">>> Prototypes initialized successfully!")

    def compute_uncertainty(self, logits):
        """
        基于 EDL 计算不确定性
        u = K / sum(alpha), where alpha = exp(logits) + 1
        """
        # evidence = torch.exp(logits)
        logits = logits / 0.5
        logits_center = logits - logits.mean(dim=1, keepdim=True)
        evidence = F.softplus(logits_center)
        evidence = torch.clamp(evidence, min=0.0, max=10.0)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = self.num_classes / S
        return uncertainty, alpha

    def forward(self, img, text_embeddings):
        """
        img: [B, 3, 224, 224]   img torch.Size([16, 3, 224, 224])
        text_embeddings: [Num_Classes, 512] (聚合后的增强文本特征)    text ViT-16=torch.Size([27, 512])
        """
        # A. 图像编码 (获取 Patch 特征)
        visual_patches = self._get_visual_features(img)
        B, L, D = visual_patches.shape  # torch.Size([16, 196, 512])

        # B. 特征解耦
        features = self.adapter(visual_patches)  # [B, L, 1024]
        delta_inv, delta_spec = torch.split(features, self.feat_dim, dim=2)  # [B, L, 512]
        v_inv = visual_patches + 0.2 * delta_inv
        # v_inv, v_spec = torch.split(features, self.feat_dim, dim=2)  # [B, L, 512]
        v_spec = delta_spec

        # C. 域对抗 (Domain Adversarial)
        v_spec_pool = v_spec.mean(dim=1)  # [B, 512]
        # disc_pred = self.discriminator(grad_reverse(v_inv_pool, alpha))
        disc_pred = self.discriminator(v_spec_pool)

        # D. 拓扑逻辑与跨模态注意力
        # Text Embeddings 作为 Key/Value, V_inv 作为 Query
        # 扩展 text 以匹配 batch: [B, Num_Classes, 512]
        k_v_text = text_embeddings.unsqueeze(0).expand(B, -1, -1)

        # Cross Attention: 图像 Patch 寻找对应的 文本描述
        # csa_out: [B, L, 512], attn_weights: [B, L, Num_Classes]
        csa_out, attn_weights = self.cross_attn(v_inv, k_v_text, k_v_text)

        # === 基于不确定性的动态加权 ===
        # 1. 先计算“纯视觉”特征的不确定性, 使用 v_inv 的均值与原型进行一次粗糙分类
        v_inv_mean = v_inv.mean(dim=1)
        f_norm_vis = F.normalize(v_inv_mean, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_vis = logit_scale * (f_norm_vis @ p_norm.t())

        # 计算视觉不确定性 (Uncertainty of Visual Features)
        u_vis, _ = self.compute_uncertainty(logits_vis)  # [B, 1]

        # 2. 动态调整权重
        # 如果不确定性 u_vis 大（图像质量差），则增大文本特征 csa_out 的权重, 基础权重设为 1.0，根据不确定性进行缩放
        w_text = 1.0 + 2.0 * u_vis.view(B, 1, 1)

        # 融合特征: v_inv (视觉) + w_text * csa_out (文本引导)
        f_aligned = v_inv + w_text * csa_out
        # f_aligned = v_inv + csa_out

        # 生成拓扑损失所需的两个 Map
        # 1. m_vis (视觉热力图): 网络自己学出来的重点区域
        m_vis = self.spatial_bridge(f_aligned)  # [B, 1, 7, 7]

        # 2. m_txt (文本引导图): 文本认为哪些 Patch 重要
        # 取每个 Patch 对所有类别的最大响应，或者对预测类别的响应(训练时用Label，测试时用Max)
        # 这里简化为 Max Response，代表“任意病害描述”激活的区域
        m_txt_raw = attn_weights.max(dim=2)[0]  # [B, 49]
        H_w = int(L ** 0.5)
        m_txt = m_txt_raw.view(B, 1, H_w, H_w).detach()  # [B, 1, 7, 7] (Detach作为GT)

        # E. 分类 (使用原型)
        # 视觉注意力加权池化
        # f_final: [B, 512]
        f_weighted = (f_aligned * m_vis.view(B, L, 1)).sum(dim=1)

        # 归一化用于余弦相似度
        f_norm = F.normalize(f_weighted, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)

        # Cosine Similarity Logits (scaled)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * (f_norm @ p_norm.t())
        # 计算最终的不确定性 (用于伪标签筛选)
        uncertainty_final, alpha_dist = self.compute_uncertainty(logits)

        return {
            "logits": logits,  # 分类结果
            "uncertainty": uncertainty_final,  # [B, 1]
            "alpha": alpha_dist,  # [B, Num_Classes] 用于 loss
            "visual_uncertainty": u_vis,  # 记录一下视觉不确定性，方便调试
            "disc_pred": disc_pred,  # 域判别结果
            "origin_visual": visual_patches,
            "v_inv": v_inv,  # 用于解耦损失
            "v_spec": v_spec,  # 用于解耦损失
            "m_vis": m_vis,  # 用于拓扑损失
            "m_txt": m_txt,  # 用于拓扑损失
            "f_weighted": f_weighted,  # 用于原型损失 (特征)
            "prototypes": self.prototypes  # 用于原型损失 (中心)
        }

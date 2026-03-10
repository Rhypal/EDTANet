import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLoss(nn.Module):
    """
    原型损失：拉近特征与对应类别原型的距离 (Cluster Compactness)
    """
    def __init__(self):
        super(PrototypeLoss, self).__init__()

    def forward(self, features, prototypes, labels):
        # features: [B, 512], prototypes: [Num_Classes, 512], labels: [B]
        if features.size(0) == 0:
            return torch.tensor(0.0).to(features.device)

        # 归一化
        f_norm = F.normalize(features, p=2, dim=1, eps=1e-6)
        p_norm = F.normalize(prototypes, p=2, dim=1, eps=1e-6)

        # 取出对应标签的原型
        target_prototypes = p_norm[labels]

        # 计算距离 (1 - Cosine Sim) 或者 Euclidean
        # loss = F.mse_loss(f_norm, target_prototypes)      # 这里用 MSE 模拟 Euclidean 距离的平方
        cos = (f_norm * target_prototypes).sum(dim=1)
        loss = (1-cos).mean()
        return loss


class GradientCorrelationLoss(nn.Module):
    """计算梯度一致性"""
    def __init__(self):
        super().__init__()
        # 定义 Sobel 算子用于计算梯度
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, map_vis, map_txt):
        g_vis_x = F.conv2d(map_vis, self.kernel_x, padding=1)
        g_vis_y = F.conv2d(map_vis, self.kernel_y, padding=1)
        g_txt_x = F.conv2d(map_txt, self.kernel_x, padding=1)
        g_txt_y = F.conv2d(map_txt, self.kernel_y, padding=1)

        grad_vis = torch.sqrt(g_vis_x ** 2 + g_vis_y ** 2 + 1e-8)
        grad_txt = torch.sqrt(g_txt_x ** 2 + g_txt_y ** 2 + 1e-8)

        flat_vis = F.normalize(grad_vis.view(grad_vis.size(0), -1), dim=1, eps=1e-6)
        flat_txt = F.normalize(grad_txt.view(grad_txt.size(0), -1), dim=1, eps=1e-6)
        # 相关性越高越好 (接近1)，损失越低
        return 1 - (flat_vis * flat_txt).sum(dim=1).mean()


class TVLoss(nn.Module):
    """
    全变分正则化 (Total Variation Regularization)
    用于去噪，使注意力图更加平滑，去除孤立的噪声点。
    """
    def forward(self, x):
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        return 2 * (h_tv + w_tv) / (x.size(0) * x.size(2) * x.size(3))


class TopologyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.grad_loss = GradientCorrelationLoss()
        self.tv_loss = TVLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, m_vis, m_txt):
        return self.alpha * self.grad_loss(m_vis, m_txt) + self.beta * self.tv_loss(m_vis)


class DecouplingLoss(nn.Module):
    """
    解耦损失：包含重构损失和正交损失
    """
    def forward(self, original, v_inv, v_spec):
        recon = v_inv + v_spec
        loss_recon = F.mse_loss(recon, original)

        v_inv_n = F.normalize(v_inv, dim=-1, eps=1e-6)
        v_spec_n = F.normalize(v_spec, dim=-1, eps=1e-6)
        loss_ortho = torch.mean(torch.abs(torch.sum(v_inv_n * v_spec_n, dim=-1)))

        return loss_recon + 0.2 * loss_ortho


class EvidenceLoss(nn.Module):
    """
    基于证据深度学习 (EDL) 的损失函数
    用于量化分类不确定性
    """
    def __init__(self, num_classes, annealing_step=10):
        super(EvidenceLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step

    def kl_divergence(self, alpha, num_classes):
        # 计算狄利克雷分布与均匀分布之间的 KL 散度
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(ones).sum(dim=1, keepdim=True)
                - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        return first_term + second_term

    def log_likelihood_loss(self, y, alpha):
        # 负对数似然损失
        S = torch.sum(alpha, dim=1, keepdim=True)
        log_likelihood = torch.sum(y * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
        return log_likelihood

    def mse_loss(self, y, alpha, epoch_num):
        # EDL-MSE Loss
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        m = alpha / S

        A = torch.sum((y - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

        # Annealing coefficient (KL散度权重的退火系数)
        annealing_coef = min(1, epoch_num / self.annealing_step)

        # KL Regularization: 迫使误分类样本的证据趋向于0（高不确定性）
        kl = annealing_coef * self.kl_divergence((alpha - 1) * (1 - y) + 1, self.num_classes)

        return torch.mean(A + B + kl)

    def forward(self, logits, targets, epoch_num):
        """
        logits: 模型输出 (未经过 softmax/activation)
        targets: 真实标签 [B]
        """
        # 将 targets 转为 One-hot
        if targets.dim() == 1:
            y = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            y = targets

        # 将 logits 转换为 Evidence (非负); 对于 Cosine Classifier，logits 范围约 [-scale, scale]。
        # evidence = torch.exp(logits)
        logits_center = logits - logits.mean(dim=1, keepdim=True)
        evidence = F.softplus(logits_center)
        evidence = torch.clamp(evidence, min=0.0, max=10.0)
        alpha = evidence + 1

        loss = self.mse_loss(y, alpha, epoch_num)
        return loss

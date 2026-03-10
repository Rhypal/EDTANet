import os
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageDraw, ImageFont
from dataset.Adataset import AgDataset
from dataset.config import PVi_PDc_common_classes
from network import DAModel
from load_text import generate_gpt_embeddings
from utils import set_seed


plt.rcParams['font.family'] = 'serif'       # 设置字体系列为 serif (衬线体)
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']   # 指定 serif 系列的首选字体为新罗马


# ================= 配置区域 (保持不变) =================
class Args:
    root_dir = '/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/Agriculture'
    checkpoint_path = '../train_output/ours/PlantVillage2PlantDoc2/best_model.pth'
    target_domain = 'PlantDoc2'
    dataset_name = 'PVi-PDc'
    json_path = '../dataset/description.json'
    clip_path = '../clip/ViT-B-16.pt'
    output_dir = '../visualization/ours_PVi-PDc'  # 换个新文件夹
    batch_size = 32
    device = 'cuda:0'
    shared_classes = sorted(list(PVi_PDc_common_classes))
    num_classes = len(shared_classes)
    top_k = 0.4


def put_text_custom(img, text, position, font_size, color):
    # 1. OpenCV (BGR) -> PIL (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 2. 加载字体
    font = ImageFont.load_default(size=font_size)

    # 3. 绘制文字 (注意：PIL 需要 RGB 颜色，所以把输入的 BGR color 翻转一下)
    draw.text(position, text, font=font, fill=color[::-1])

    # 4. PIL (RGB) -> OpenCV (BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ================= 改进的可视化工具 =================
def denormalize(tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
    img = tensor.clone().detach().permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def features_to_heatmap(features, target_size=(224, 224), colormap=cv2.COLORMAP_JET, power=4):
    """
    改进版：增加阈值过滤，减少背景噪声
    """
    # 1. 维度处理
    if features.dim() == 2:  # [L, D]
        L, D = features.shape
        H = int(L ** 0.5)
        if H * H == L and D > H:  # ViT Features
            features = features.view(H, H, D)
            heatmap = features.mean(dim=2)  # 均值聚合
        else:
            heatmap = features  # 已经是2D map
    elif features.dim() == 3:  # [C, H, W]
        heatmap = features.mean(dim=0)
    else:
        heatmap = features

    heatmap = heatmap.detach().cpu().numpy()

    # 2. 归一化 [0, 1]
    min_val, max_val = heatmap.min(), heatmap.max()
    if max_val - min_val > 1e-8:
        heatmap = (heatmap - min_val) / (max_val - min_val)
    else:
        heatmap = np.zeros_like(heatmap)

    # === 关键改进：过滤掉低响应区域 ===
    heatmap = heatmap ** power
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-7)

    # 4. 缩放
    heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)

    # 5. 伪彩色
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    return heatmap_color


def create_composite(original, feat_map, attn_map, separator_width=5):
    h, w, _ = original.shape
    separator = np.zeros((h, separator_width, 3), dtype=np.uint8)
    composite = np.concatenate((original, separator, feat_map, separator, attn_map), axis=1)
    return composite


# ================= 主程序 =================
def main():
    set_seed(42)
    os.makedirs(Args.output_dir, exist_ok=True)
    device = torch.device(Args.device)

    print(">>> Loading Test Dataset...")
    test_ds = AgDataset(Args.root_dir, Args.target_domain, Args.shared_classes, is_source=False, phase='test')
    test_loader = DataLoader(test_ds, batch_size=Args.batch_size, shuffle=False, num_workers=4)

    print(">>> Generating Text Embeddings...")
    text_embeddings = generate_gpt_embeddings(Args.dataset_name, Args.shared_classes, Args.json_path, Args.clip_path,
                                              device, Args.top_k).detach()

    print(">>> Initializing Model...")
    model = DAModel(Args.num_classes, clip_weight_path=Args.clip_path, device=device).to(device)

    if os.path.exists(Args.checkpoint_path):
        checkpoint = torch.load(Args.checkpoint_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict, strict=True)
        print("Weights loaded.")
    else:
        return

    model.eval()
    idx_to_class = {i: name for i, name in enumerate(Args.shared_classes)}

    print(">>> Running Inference...")
    with torch.no_grad():
        for batch_idx, (imgs, labels, _) in enumerate(tqdm(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward (注意：这里必须确保 DAModel 返回 attn_weights)
            out = model(imgs, text_embeddings)

            logits = out['logits']
            origin_visual = out['origin_visual']  # [B, 196, 512]

            # === 1. 获取 Cross-Attention 权重 ===
            # [B, 196, Num_Classes]
            attn_weights = out['attn_weights']
            if attn_weights is None:
                print("Error: Model output does not contain 'attn_weights'. Check network.py")
                break

            _, preds = torch.max(logits, 1)

            loop_limit = min(imgs.size(0), 3)
            for b in range(loop_limit):
                img_rgb = denormalize(imgs[b])

                # --- Map 1: 原始特征 (通常比较散乱，作为对比) ---
                # 不过滤弱信号，展示原始分布
                feat_heatmap = features_to_heatmap(origin_visual[b])
                feat_vis = cv2.addWeighted(img_rgb, 0.6, feat_heatmap, 0.4, 0)

                # --- Map 2: 类别特定的注意力 (Class-Specific Attention) ---
                # 关键步骤：提取预测类别对应的注意力列
                pred_class_idx = preds[b].item()

                # [196, Num_Classes] -> 取第 pred_class_idx 列 -> [196]
                specific_attn = attn_weights[b, :, pred_class_idx]

                # Reshape 成 [14, 14]
                H = int(196 ** 0.5)
                specific_attn_map = specific_attn.view(H, H)

                # 生成热力图 (开启过滤，只看强关注点)
                attn_heatmap = features_to_heatmap(specific_attn_map, power=3)

                # 叠加 (热力图稍微强一点)
                attn_vis = cv2.addWeighted(img_rgb, 0.6, attn_heatmap, 0.4, 0)

                # --- 拼接与保存 ---
                composite = create_composite(img_rgb, feat_vis, attn_vis)

                gt_name = idx_to_class[labels[b].item()]
                pred_name = idx_to_class[preds[b].item()]
                is_correct = labels[b] == preds[b]
                status = "RIGHT" if is_correct else "WRONG"

                # Header
                pad_height = 40
                header = np.zeros((pad_height, composite.shape[1], 3), dtype=np.uint8) + 255
                color = (0, 0, 0) if is_correct else (0, 0, 255)  # 绿对红错
                info_text = f"ours | GT: {gt_name} | Pred: {pred_name}"
                header = put_text_custom(img=header, text=info_text, position=(10, 10), font_size=17, color=color)
                final_img = np.vstack([header, composite])

                filename = f"Batch{batch_idx}_Img{b}_{status}_GT-{gt_name}.jpg"
                filename = filename.replace(' ', '_').replace('/', '-')
                cv2.imwrite(os.path.join(Args.output_dir, filename), final_img)

    print(f"Done! Saved to {Args.output_dir}")


if __name__ == '__main__':
    main()

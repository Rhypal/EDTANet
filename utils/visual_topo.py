import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from tqdm import tqdm
# === 导入你的自定义模块 ===
from dataset.Adataset import AgDataset
from dataset.config import PVi_PDc_common_classes  # 确保这里和你的训练类别一致
from network import DAModel
from load_text import generate_gpt_embeddings


plt.rcParams['font.family'] = 'serif'       # 设置字体系列为 serif (衬线体)
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']   # 指定 serif 系列的首选字体为新罗马

class VisConfig:
    root_dir = '/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/Agriculture'
    target_domain = 'PlantDoc'  # 我们通常可视化目标域测试集
    dataset_name = 'PVi-PDc'
    json_path = '../dataset/description.json'
    clip_path = '../clip/ViT-B-16.pt'

    # === 需要你修改的权重路径 ===
    model_weight_path = '../train_output/ours/PlantVillage2PlantDoc/best_model.pth'
    save_dir = '../visualization/topology'  # 图片保存位置

    device = 'cuda:0'
    shared_classes = sorted(list(PVi_PDc_common_classes))
    num_classes = len(shared_classes)
    # === 指定你想可视化的特定类别 ===
    target_class_name = 'Apple_Scab'  # 替换成你实际想看的类别名
    num_vis_samples = 303  # 指定该类别最多保存多少张（填 9999 可以保存该类全部）


def denormalize(tensor, device):
    """把 CLIP 的归一化参数反转回来，恢复原始图像供可视化"""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)
    img = tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return img


def apply_heatmap(img_rgb, attention_map):
    """
    将 7x7 的特征图放大，使用 JET (红高蓝低) 色带，并进行动态透明度融合。
    """
    att_map = attention_map.detach().cpu().numpy()
    att_map = np.maximum(att_map, 0)  # ReLU 过滤负值

    # === 核心修复 1：使用 Min-Max 归一化 ===
    # 强制将注意力值拉伸到 0.0 ~ 1.0 的满量程。
    # 这样才能保证最低的响应绝对是蓝色，最高的响应绝对是红色。
    min_val = att_map.min()
    max_val = att_map.max()
    if max_val > min_val:
        att_map = (att_map - min_val) / (max_val - min_val)
    elif max_val > 0:
        att_map = att_map / max_val

    # 上采样到与原图相同大小 (224x224)，使用 BICUBIC 插值让边缘平滑
    att_map_resized = cv2.resize(att_map, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)

    # === 核心修复 2：使用标准 jet 色带 (0=深蓝, 0.5=黄绿, 1=深红) ===
    cmap = plt.get_cmap('jet')
    heatmap_rgba = cmap(att_map_resized)  # 映射出 [H, W, 4] 的颜色矩阵
    heatmap_rgb = heatmap_rgba[..., :3]  # 取出 RGB 通道

    # === 核心修复 3：动态透明度 (Dynamic Alpha Blending) ===
    # 让注意力越低的地方（趋近0）Alpha越小（趋近0.1，近乎透明的淡蓝色）；越高的地方（趋近1）Alpha越大（趋近0.6，明显的红色覆盖）。
    power_factor = 2.0  # 你可以调节这个值：越大，红斑越聚集越小；越小，光晕散得越开
    alpha = np.power(att_map_resized, power_factor)
    # alpha = 0.1 + 0.5 * att_map_resized
    alpha = alpha[..., np.newaxis]  # 扩展维度以进行广播: [224, 224, 1]

    # 真正的图层混合 (Alpha Blending)
    overlay = img_rgb * (1.0 - alpha) + heatmap_rgb * alpha
    overlay = np.clip(overlay, 0, 1)  # 确保像素值不越界

    return overlay


def main():
    os.makedirs(VisConfig.save_dir, exist_ok=True)
    device = VisConfig.device

    print(">>> 1. Loading Text Embeddings...")
    text_embeddings = generate_gpt_embeddings(
        dataset_name=VisConfig.dataset_name,
        class_names=VisConfig.shared_classes,
        json_file=VisConfig.json_path,
        clip_weight_path=VisConfig.clip_path,
        device=device,
        top_k_ratio=0.4
    ).detach()

    print(">>> 2. Initializing Model and Loading Weights...")
    model = DAModel(VisConfig.num_classes, VisConfig.clip_path, device).to(device)

    # 加载训练好的权重
    checkpoint = torch.load(VisConfig.model_weight_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()  # 切换到测试模式，关闭 Dropout 和 BatchNorm 等

    print(">>> 3. Loading Test Dataset...")
    # 使用 batch_size=1 方便逐张图片保存
    test_ds = AgDataset(VisConfig.root_dir, VisConfig.target_domain, VisConfig.shared_classes, is_source=False,
                        phase='test')
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=2)

    print(f">>> 4. Starting Visualization for class: [{VisConfig.target_class_name}]...")

    count = 0
    with torch.no_grad():
        # 使用 tqdm 包装 test_loader 显示进度
        for imgs, labels, img_paths in tqdm(test_loader, desc=f"Generating Heatmaps"):

            # 获取当前图片的类别名
            class_name = VisConfig.shared_classes[labels.item()]

            # === 核心拦截逻辑：如果不是目标类别，直接跳过 ===
            if class_name != VisConfig.target_class_name:
                continue

            # 检查是否达到了该类别的保存数量上限
            if count >= VisConfig.num_vis_samples:
                break

            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward 推理
            out = model(imgs, text_embeddings)

            # 提取两个 map (batch_size=1)
            m_vis = out['m_vis'][0, 0]  # [7, 7]
            m_txt = out['m_txt'][0, 0]  # [7, 7]

            # 为当前类别创建专属文件夹
            class_save_dir = os.path.join(VisConfig.save_dir, class_name)
            os.makedirs(class_save_dir, exist_ok=True)

            base_name = f"sample_{count:04d}"

            # --- 图像后处理 ---
            img_rgb_norm = denormalize(imgs[0], device)
            vis_overlay = apply_heatmap(img_rgb_norm, m_vis)
            txt_overlay = apply_heatmap(img_rgb_norm, m_txt)

            # --- 绘图与保存 ---
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img_rgb_norm)
            axes[0].set_title(f"Original\n({class_name})", fontsize=14)
            axes[0].axis('off')

            axes[1].imshow(vis_overlay)
            axes[1].set_title("Visual Attention (m_vis)", fontsize=14)
            axes[1].axis('off')

            axes[2].imshow(txt_overlay)
            axes[2].set_title("Text Guidance (m_txt)", fontsize=14)
            axes[2].axis('off')

            plt.tight_layout()

            save_name = f"{base_name}.png"
            save_path = os.path.join(class_save_dir, save_name)

            plt.savefig(save_path, bbox_inches='tight', dpi=300)

            plt.clf()
            plt.close(fig)

            count += 1  # 只有成功保存了目标类别的图片，计数器才 +1

    print(f">>> Done! Saved {count} images for class '{VisConfig.target_class_name}'.")


if __name__ == '__main__':
    main()

import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
# === 引入你的项目模块 ===
from dataset.Adataset import AgDataset_res
from dataset.config import PVi_PDc_common_classes
from utils import set_seed, ModelPredictor
# 导入模型定义
from comparative.DAN.mmd import DANNet
from comparative.DANN.net2 import DANN
from comparative.DAAN.net3 import DAANNet
from comparative.DSAN.net4 import DSAN
from comparative.ToAlign.net5 import ToAlignNet
from comparative.MemSAC.net6 import MemSACNet
from comparative.MSUN.net7 import MSUNNet
from comparative.MViTs.net8 import MViTs
from comparative.DG_PLDR.net9 import DGNet
from comparative.FBR.net10 import CDAN


plt.rcParams['font.family'] = 'serif'       # 设置字体系列为 serif (衬线体)
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']   # 指定 serif 系列的首选字体为新罗马


# ================= 配置区域 =================
class Args:
    root_dir = '/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/Agriculture'
    target_domain = 'PlantDoc2'
    shared_classes = sorted(list(PVi_PDc_common_classes))
    num_classes = len(shared_classes)

    # === 在这里配置你要测试的模型 ===
    method = 'FBR'     #  DAN, DANN, DAAN, DSAN, ToAlign, MemSAC, MSUN, MViTs, DG, FBR
    checkpoint_path = '../train_output/FBR_CDAN/FBR_PVi_PDc2PlantDoc/best_FBR.pth'

    output_dir = f'../visualization/PVi-PDc_{method}'
    device = 'cuda:0'
    batch_size = 32


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


# ================= 通用特征提取钩子 (核心部分) =================
class FeatureHook:
    """
    自动寻找模型的最后一个卷积层，并提取特征图。
    不需要修改模型源码，即插即用。
    """

    def __init__(self, model):
        self.model = model
        self.feature = None
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        # 倒序遍历模块，寻找最后一个 Conv2d
        # ResNet/VGG/CNN 类模型通用
        target_layer = None
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                print(f">>> Auto-Hooked layer: {name}")
                break

        if target_layer is None:
            print("Warning: No Conv2d layer found! Visualization might fail.")
            return

        # 注册 Forward Hook
        self.hook_handle = target_layer.register_forward_hook(self.save_feature)

    def save_feature(self, module, input, output):
        # output 就是该层的特征图 [B, C, H, W]
        self.feature = output

    def get_last_feature(self):
        return self.feature

    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()


# ================= 可视化工具函数 =================
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = tensor.clone().detach().permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def feature_to_heatmap(feature, target_size=(224, 224)):
    """
    将 [C, H, W] 特征图 -> [H, W] 热力图
    """
    # 1. 通道维求均值 (或者取最大值)
    if feature.dim() == 3:
        heatmap = torch.mean(feature, dim=0)  # [H, W]
    else:
        heatmap = feature

    # 2. 归一化
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU 效果，只看正激活
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 3. 缩放
    heatmap = cv2.resize(heatmap, target_size)

    # 4. 伪彩色
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return heatmap_color


def create_composite(original, heatmap, separator_width=10):
    """ 原图 | 黑条 | 特征热力图 """
    h, w, c = original.shape
    # 混合显示：原图+热力图叠加
    overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

    separator = np.zeros((h, separator_width, c), dtype=np.uint8)
    # 拼接: 原图 | 黑条 | 叠加图
    return np.hstack([original, separator, overlay])


# ================= 主程序 =================
def main():
    set_seed(42)
    os.makedirs(Args.output_dir, exist_ok=True)
    device = torch.device(Args.device)

    # 1. 加载数据
    print(f"Loading Dataset: {Args.target_domain}")
    test_ds = AgDataset_res(Args.root_dir, Args.target_domain, Args.shared_classes, is_source=False, phase='test')
    test_loader = DataLoader(test_ds, batch_size=Args.batch_size, shuffle=False, num_workers=4)
    idx_to_class = {i: name for i, name in enumerate(Args.shared_classes)}

    # 2. 加载模型 (工厂模式)
    print(f"Initializing Model: {Args.method}")
    if Args.method == 'DSAN':
        raw_model = DSAN(num_classes=Args.num_classes, weight_path=None)
    elif Args.method == 'DAN':
        raw_model = DANNet(num_classes=Args.num_classes, weight_path=None)
    elif Args.method == 'DANN':
        raw_model = DANN(num_classes=Args.num_classes, pretrain_path=None)
    elif Args.method == 'DAAN':
        raw_model = DAANNet(num_classes=Args.num_classes, pretrain_path=None)
    elif Args.method == 'ToAlign':
        raw_model = ToAlignNet(num_classes=Args.num_classes, pretrain_path=None)
    elif Args.method == 'MemSAC':
        raw_model = MemSACNet(num_classes=Args.num_classes, pretrain_path=None)
    elif Args.method == 'MSUN':
        raw_model = MSUNNet(num_classes=Args.num_classes, pretrained_path='../comparative/resnet50-19c8e357.pth')
    elif Args.method == 'MViTs':
        raw_model = MViTs(num_classes=Args.num_classes, pretrained=False)
    elif Args.method == 'DG':
        raw_model = DGNet(num_classes=Args.num_classes, pretrain_path=None)
    elif Args.method == 'FBR':
        raw_model = CDAN(num_classes=Args.num_classes, pretrained_path=None)
    else:
        raise ValueError("Unknown model method name")

    # 加载权重
    if os.path.exists(Args.checkpoint_path):
        state_dict = torch.load(Args.checkpoint_path, map_location=device)
        raw_model.load_state_dict(state_dict, strict=False)
        print("Weights loaded.")
    else:
        print(f"Warning: Checkpoint not found at {Args.checkpoint_path}")

    # 3. 封装 Predictor 和 Hook
    # (A) 用于推理的封装
    predictor = ModelPredictor(raw_model, Args.method).to(device)
    predictor.eval()

    # (B) 用于可视化的 Hook (直接挂在原始 raw_model 上),  找到最后一个 Conv 层并监听它
    viz_hook = FeatureHook(raw_model)

    # 4. 推理循环
    print(">>> Start Visualization...")
    count = 0

    with torch.no_grad():
        for batch_idx, (imgs, labels, _) in enumerate(tqdm(test_loader)):
            imgs = imgs.to(device)

            # === Forward ===
            # 这里调用 predictor 会触发 raw_model 的 forward/predict
            # 同时 viz_hook 会自动捕获中间特征
            logits = predictor(imgs)
            _, preds = torch.max(logits, 1)

            # 获取捕获的特征 [B, C, H, W]
            feature_maps = viz_hook.get_last_feature()

            # 如果没捕获到 (例如全是全连接层), 跳过
            if feature_maps is None: continue

            current_batch_size = imgs.size(0)
            limit = min(current_batch_size, 3)  # 每个Batch只看前3张

            for b in range(limit):
                # 1. 还原原图
                img_rgb = denormalize(imgs[b])

                # 2. 处理特征图
                # [C, H, W] -> Heatmap
                feat_map = feature_to_heatmap(feature_maps[b], target_size=(224, 224))

                # 3. 拼接
                composite = create_composite(img_rgb, feat_map, separator_width=10)

                # 4. 绘制标签
                gt_name = idx_to_class[labels[b].item()]
                pred_name = idx_to_class[preds[b].item()]
                is_correct = (labels[b] == preds[b])

                # 头部加白条写字
                header = np.zeros((40, composite.shape[1], 3), dtype=np.uint8) + 255
                color = (0, 0, 0) if is_correct else (0, 0, 255)  # 绿对(0, 255, 0)红错(0, 0, 255)
                text_content = f"[{Args.method}] GT:{gt_name} | Pred:{pred_name}"

                # === 修改：使用自定义函数替代 cv2.putText ===
                # position=(10, 10): 这里的 y=10 是文字顶端距离 header 顶部的像素，放在 y=10 刚好居中
                header = put_text_custom(img=header, text=text_content, position=(10, 10), font_size=16, color=color)

                final_img = np.vstack([header, composite])

                # 5. 保存
                status = "RIGHT" if is_correct else "WRONG"
                filename = f"Batch{batch_idx}_Img{b}_{status}_GT-{gt_name}.jpg"
                filename = filename.replace(' ', '_').replace('/', '-')
                cv2.imwrite(os.path.join(Args.output_dir, filename), final_img)

                count += 1

    # 清理 Hook
    viz_hook.remove()
    print(f"Done! Saved to {Args.output_dir}")


if __name__ == '__main__':
    main()

import os
import random
import time

import numpy as np
import torch
import logging
from datetime import datetime
from torch.utils.data import DataLoader


def get_logger(output_dir):
    """创建日志记录器，同时输出到控制台和txt文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用时间戳命名日志文件，防止覆盖
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'train_log_{time_str}.txt')

    def log_string(out_str):
        print(out_str)
        with open(log_file, 'a') as f:
            f.write(out_str + '\n')

    return log_string


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelPredictor:
    def __init__(self, model, method_name):
        """
        Args:
            model: 实际的 PyTorch 模型实例
            method_name: 方法名 (e.g., 'DAN', 'DANN')
        """
        self.model = model
        self.method_name = method_name

    def __call__(self, x):
        """
        让类的实例可以像函数一样被调用: output = wrapper(imgs)
        统一返回: Logits (预测结果)
        """
        if self.method_name in ['DAN', 'DSAN']:     # logits, loss
            return self.model.predict(x)

        elif self.method_name == 'DANN':
            output, domain = self.model(x, alpha=0)  # 测试时 alpha 通常设为 0 或 None
            return output

        elif self.method_name == 'DAAN':
            output, domain_g, domain_l = self.model(x, alpha=0)
            return output

        #  ToAlign / MemSAC / MViTs: 返回 (features, output)
        elif self.method_name in ['ToAlign', 'MemSAC', 'MViTs']:
            feature, output = self.model(x)
            return output

        elif self.method_name == 'MSUN':
            output = self.model(x, None)
            return output

        elif self.method_name == 'DG':
            output, feature = self.model(x, None, None)
            return output

        elif self.method_name == 'FBR':
            output, domain = self.model(x, alpha=0, use_entropy=False)
            return output

        # 7. 默认情况 (ResNet, EfficientNet 等普通模型)
        else:
            return self.model(x)

    def eval(self):
        """ 透传 eval 调用 """
        self.model.eval()

    def train(self):
        """ 透传 train 调用 """
        self.model.train()

    def to(self, device):
        """ 透传 to(device) 调用 """
        self.model.to(device)
        return self  # 支持链式调用

    # (可选) 如果需要访问原始模型的属性，使用 getattr 自动转发
    def __getattr__(self, name):
        return getattr(self.model, name)


def test_ours(model, loader, text_embeddings, device, class_names, logger=None):
    model.eval()

    # === 初始化统计字典 ===
    plant_stats = {}
    idx_to_plant = {}
    for idx, name in enumerate(class_names):
        plant_name = name.split('_')[0]
        idx_to_plant[idx] = plant_name
        if plant_name not in plant_stats:
            plant_stats[plant_name] = {'correct': 0, 'total': 0}

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs, text_embeddings)
            _, preds = torch.max(output['logits'], 1)

            for i in range(len(labels)):
                label_idx = labels[i].item()
                pred_idx = preds[i].item()
                plant_name = idx_to_plant[label_idx]

                plant_stats[plant_name]['total'] += 1
                total_samples += 1
                if label_idx == pred_idx:
                    plant_stats[plant_name]['correct'] += 1
                    total_correct += 1

    # === 构建输出字符串 ===
    # 将所有输出拼接成一个长字符串，而不是一行行打印
    plant_names_sorted = sorted(plant_stats.keys())

    header_str = "| {:<12} ".format("Method")
    data_str = "| {:<12} ".format("ours")

    macro_avg_sum = 0
    valid_plants = 0

    for plant in plant_names_sorted:
        stats = plant_stats[plant]
        if stats['total'] > 0:
            acc = (stats['correct'] / stats['total']) * 100
            macro_avg_sum += acc
            valid_plants += 1
            header_str += "| {:<10} ".format(plant)
            data_str += "| {:<10.2f} ".format(acc)

    avg_acc = macro_avg_sum / valid_plants if valid_plants > 0 else 0.0

    header_str += "| {:<10} |".format("Average")
    data_str += "| {:<10.2f} |".format(avg_acc)

    separator = "-" * len(header_str)

    # 组合成完整的表格字符串
    table_msg = (
        f"\n{separator}\n"
        f"{header_str}\n"
        f"{separator}\n"
        f"{data_str}\n"
        f"{separator}\n"
    )

    # === 输出 ===
    # 如果传入了 logger，就用 logger 记录（这通常会写入文件）
    if logger:
        logger(table_msg)
    else:
        # 如果没传 logger，为了防止没输出，还是打印一下
        print(table_msg)

    return total_correct / total_samples


def test_per_class(model, loader, device, class_names, logger=None, text_embeddings=None):
    model.eval()

    # === 修改 1: 初始化统计字典 (按具体类别统计) ===
    # 使用完整类名作为 key
    class_stats = {name: {'correct': 0, 'total': 0} for name in class_names}

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs, text_embeddings)     # ours
            _, preds = torch.max(output['logits'], 1)

            # outputs = model(imgs)       # resnet
            # _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                label_idx = labels[i].item()
                pred_idx = preds[i].item()

                # === 修改 2: 获取具体类别名称 ===
                label_name = class_names[label_idx]

                class_stats[label_name]['total'] += 1
                total_samples += 1

                if label_idx == pred_idx:
                    class_stats[label_name]['correct'] += 1
                    total_correct += 1

    # === 修改 3: 构建输出格式 (改为纵向列表，防止横向太长) ===
    lines = []
    lines.append("-" * 65)
    # 表头: 类名 | 精度 | 样本数
    lines.append(f"| {'Class Name':<35} | {'Acc (%)':<10} | {'Count':<6} |")
    lines.append("-" * 65)

    macro_avg_sum = 0
    valid_classes = 0

    # 按名称排序输出
    for name in sorted(class_names):
        stats = class_stats[name]
        if stats['total'] > 0:
            acc = (stats['correct'] / stats['total']) * 100
            macro_avg_sum += acc
            valid_classes += 1
            # 格式化每一行
            lines.append(f"| {name:<35} | {acc:<10.2f} | {stats['total']:<6} |")
        else:
            # 如果某类没有样本
            lines.append(f"| {name:<35} | {'N/A':<10} | {0:<6} |")

    lines.append("-" * 65)

    # 计算平均值
    avg_acc = macro_avg_sum / valid_classes if valid_classes > 0 else 0.0
    overall_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

    # 添加汇总信息
    lines.append(f"| {'Macro Average (Per Class)':<35} | {avg_acc:<10.2f} | {'-':<6} |")
    lines.append(f"| {'Overall Accuracy':<35} | {overall_acc:<10.2f} | {total_samples:<6} |")
    lines.append("-" * 65)

    # 组合成完整的字符串
    table_msg = "\n" + "\n".join(lines) + "\n"

    # === 输出 ===
    if logger:
        logger(table_msg)
    else:
        print(table_msg)

    # 返回总体精度 (0-1 float)
    return total_correct / total_samples


def test(model, loader, device, class_names, logger=None):
    model.eval()

    # === 初始化统计字典 ===
    plant_stats = {}
    idx_to_plant = {}
    for idx, name in enumerate(class_names):
        plant_name = name.split('_')[0]
        idx_to_plant[idx] = plant_name
        if plant_name not in plant_stats:
            plant_stats[plant_name] = {'correct': 0, 'total': 0}

    total_correct = 0
    total_samples = 0

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == 'cuda':       # 确保 GPU 之前的任务都做完了
        torch.cuda.synchronize()
    start_time = time.time()        # 记录开始时间

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            _, preds = torch.max(output, 1)

            for i in range(len(labels)):
                label_idx = labels[i].item()
                pred_idx = preds[i].item()
                plant_name = idx_to_plant[label_idx]

                plant_stats[plant_name]['total'] += 1
                total_samples += 1
                if label_idx == pred_idx:
                    plant_stats[plant_name]['correct'] += 1
                    total_correct += 1

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()      # 记录结束时间
    total_time = end_time - start_time
    fps = total_samples / total_time        # 每秒处理图片数

    # === 构建输出字符串 ===
    # 将所有输出拼接成一个长字符串，而不是一行行打印
    plant_names_sorted = sorted(plant_stats.keys())

    header_str = "| {:<12} ".format("Method")
    data_str = "| {:<12} ".format("ours")

    macro_avg_sum = 0
    valid_plants = 0

    for plant in plant_names_sorted:
        stats = plant_stats[plant]
        if stats['total'] > 0:
            acc = (stats['correct'] / stats['total']) * 100
            macro_avg_sum += acc
            valid_plants += 1
            header_str += "| {:<10} ".format(plant)
            data_str += "| {:<10.2f} ".format(acc)

    avg_acc = macro_avg_sum / valid_plants if valid_plants > 0 else 0.0

    header_str += "| {:<10} |".format("Average")
    data_str += "| {:<10.2f} |".format(avg_acc)

    separator = "-" * len(header_str)

    # 组合成完整的表格字符串
    table_msg = (
        f"\n{separator}\n"
        f"{header_str}\n"
        f"{separator}\n"
        f"{data_str}\n"
        f"{separator}\n"
    )

    time_msg = (
        f"Total Samples: {total_samples}\n"
        f"Total Time:    {total_time:.4f} sec\n"
        f"FPS:           {fps:.2f} img/sec\n"
    )

    # === 输出 ===
    # 如果传入了 logger，就用 logger 记录（这通常会写入文件）
    if logger:
        logger(table_msg)
    else:
        # 如果没传 logger，为了防止没输出，还是打印一下
        print(table_msg)

    return total_correct / total_samples



def test_dg(model, loader, device, class_names, logger=None):
    model.eval()

    # === 初始化统计字典 ===
    plant_stats = {}
    idx_to_plant = {}
    for idx, name in enumerate(class_names):
        plant_name = name.split('_')[0]
        idx_to_plant[idx] = plant_name
        if plant_name not in plant_stats:
            plant_stats[plant_name] = {'correct': 0, 'total': 0}

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output, _ = model(imgs, None, None)       # DG-PLDR
            _, preds = torch.max(output, 1)

            for i in range(len(labels)):
                label_idx = labels[i].item()
                pred_idx = preds[i].item()
                plant_name = idx_to_plant[label_idx]

                plant_stats[plant_name]['total'] += 1
                total_samples += 1
                if label_idx == pred_idx:
                    plant_stats[plant_name]['correct'] += 1
                    total_correct += 1

    # === 构建输出字符串 ===
    # 将所有输出拼接成一个长字符串，而不是一行行打印
    plant_names_sorted = sorted(plant_stats.keys())

    header_str = "| {:<12} ".format("Method")
    data_str = "| {:<12} ".format("ours")

    macro_avg_sum = 0
    valid_plants = 0

    for plant in plant_names_sorted:
        stats = plant_stats[plant]
        if stats['total'] > 0:
            acc = (stats['correct'] / stats['total']) * 100
            macro_avg_sum += acc
            valid_plants += 1
            header_str += "| {:<10} ".format(plant)
            data_str += "| {:<10.2f} ".format(acc)

    avg_acc = macro_avg_sum / valid_plants if valid_plants > 0 else 0.0

    header_str += "| {:<10} |".format("Average")
    data_str += "| {:<10.2f} |".format(avg_acc)

    separator = "-" * len(header_str)

    # 组合成完整的表格字符串
    table_msg = (
        f"\n{separator}\n"
        f"{header_str}\n"
        f"{separator}\n"
        f"{data_str}\n"
        f"{separator}\n"
    )

    # === 输出 ===
    # 如果传入了 logger，就用 logger 记录（这通常会写入文件）
    if logger:
        logger(table_msg)
    else:
        # 如果没传 logger，为了防止没输出，还是打印一下
        print(table_msg)

    return total_correct / total_samples

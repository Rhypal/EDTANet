import os
import torch
from clip import clip
import json
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch.nn.functional as F


def attr_clustering(
        dataset_name='PlantVillage',
        num_attr_clusters=64,
        json_file='./dataset/description.json',
        clip_weight_path='./clip/ViT-B-16.pt',
        selected_classes=None,
        device='cuda'
):
    suffix = "_shared" if selected_classes else "_full"
    save_dir = './dataset'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{dataset_name}{suffix}_cluster_{num_attr_clusters}.pth')

    if os.path.exists(save_path):
        print(f">> 加载缓存的属性库: {save_path}")
        return torch.load(save_path)

    print(f"正在加载描述文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        full_descriptions = json.load(f)

    # 过滤类别
    if selected_classes:
        descriptions = {k: v for k, v in full_descriptions.items() if k in selected_classes}
    else:
        descriptions = full_descriptions

    # 加载 CLIP
    print(f"Loading CLIP: {clip_weight_path}")
    model, _ = clip.load(clip_weight_path, device=device, jit=False)
    model.eval()

    # 提取特征
    all_embeddings = []
    print("正在提取文本特征...")
    for cls in tqdm(sorted(descriptions.keys())):
        sentences = [s.lower() for s in descriptions[cls]]
        tokens = clip.tokenize(sentences, truncate=True).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=1, keepdim=True)
            all_embeddings.append(feats.cpu())

    all_tensor = torch.cat(all_embeddings, dim=0)

    # 聚类
    real_k = min(num_attr_clusters, all_tensor.shape[0])
    print(f"正在聚类: {all_tensor.shape[0]} 条描述 -> {real_k} 个属性中心")
    kmeans = KMeans(n_clusters=real_k, random_state=42, n_init=10).fit(all_tensor.numpy())

    # 属性库: [512, K] (转置后方便矩阵乘法)
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).t()

    torch.save([cluster_centers, None], save_path)
    return [cluster_centers, None]


def attr_aggregate(class_text_features, attribute_bank, topK=0.5):
    """
    class_text_features: [Num_Classes, 512]
    attribute_bank: [512, N_clusters]
    """
    device = class_text_features.device
    attr_features = attribute_bank.to(device)

    # 计算相关性 [N_cls, N_clusters]
    logits = class_text_features @ attr_features

    # Top-K 过滤
    if topK is not None:
        k = int((1 - topK) * attr_features.shape[1])
        if k > 0:
            values, indices = torch.topk(logits, k=attr_features.shape[1] - k, dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, indices, values)
            logits = mask

    attn_weights = logits.softmax(dim=-1)

    # 聚合: [N_cls, N_clusters] @ [N_clusters, 512]
    enhanced = class_text_features + (attn_weights @ attr_features.t())
    return enhanced / enhanced.norm(dim=1, keepdim=True)


def generate_gpt_embeddings(
        dataset_name='PlantVillage',
        class_names=None,  # 必填：传入 Args.shared_classes
        json_file='./dataset/description.json',
        clip_weight_path='./clip/ViT-B-16.pt',
        device='cuda:0',
        top_k_ratio=0.6,  # 核心参数：只保留相似度最高的 60% 描述，剔除 40% 的噪声/幻觉
        save_dir='./dataset'
):
    """
    功能：
    1. 读取每个类别的 GPT 描述。
    2. 计算该类别所有描述的平均中心。
    3. 筛选：保留与中心最接近的 top_k_ratio 部分描述（去噪）。
    4. 聚合：计算筛选后描述的均值，作为该类别的最终 Text Embedding。
    返回：
    text_embeddings: [Num_Classes, 512]
    """
    # 1. 定义缓存路径 (加入 ratio 防止参数变化后读取旧缓存)
    if class_names is None:
        raise ValueError("必须传入 class_names (Args.shared_classes) 以确保顺序一致")

    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_k{int(top_k_ratio * 100)}"
    save_path = os.path.join(save_dir, f'{dataset_name}_gpt_cluster{suffix}.pth')

    # 2. 尝试加载缓存
    if os.path.exists(save_path):
        print(f">> [Info] 加载缓存的 GPT 文本原型: {save_path}")
        return torch.load(save_path, map_location=device)

    # 3. 开始重新生成
    print(f">> [Info] 正在生成文本原型... (Source: {json_file})")
    print(f">> [Config] Top-K Ratio: {top_k_ratio} (剔除 {100 - int(top_k_ratio * 100)}% 的离群描述)")

    with open(json_file, 'r', encoding='utf-8') as f:
        full_descriptions = json.load(f)

    # 加载 CLIP (如果外部没有加载，这里加载；为了省显存建议外部传模型进来，这里简化处理)
    # 注意：为了防止多次加载模型占用显存，建议在函数外加载好 clip_model 传进来
    # 这里为了保持你原代码的独立性，还是在内部加载
    print(f"Loading CLIP: {clip_weight_path}")
    model, _ = clip.load(clip_weight_path, device=device, jit=False)
    model.eval()

    class_prototypes = []

    # 4. 逐类别处理 (保证顺序与 class_names 一致)
    for cls_name in tqdm(class_names, desc="Processing Classes"):
        # --- A. 获取描述 ---
        # 容错处理：处理可能的 key 不匹配问题 (如 JSON 中是 Apple___Scab 但 class_names 是 Apple Scab)
        if cls_name in full_descriptions:
            texts = full_descriptions[cls_name]
        else:
            # 尝试把空格换成下划线，或者反之
            alt_name = cls_name.replace(' ', '_')
            if alt_name in full_descriptions:
                texts = full_descriptions[alt_name]
            else:
                print(f"\n[Warning] 没找到类别 '{cls_name}' 的描述! 使用默认 Prompt 兜底。")
                texts = [f"A photo of {cls_name}, a plant disease."]

        # --- B. 编码所有描述 ---
        with torch.no_grad():
            # [N_descs, 77]
            tokens = clip.tokenize(texts, truncate=True).to(device)
            # [N_descs, 512]
            feats = model.encode_text(tokens).float()
            feats = F.normalize(feats, dim=1)

        # --- C. 聚类/筛选核心逻辑 (Mean Shift 思想) ---
        if feats.shape[0] > 1:
            # 1. 计算粗略中心
            rough_center = feats.mean(dim=0, keepdim=True)  # [1, 512]
            rough_center = F.normalize(rough_center, dim=1)

            # 2. 计算每条描述与中心的相似度
            # [N_descs]
            sims = (feats @ rough_center.t()).squeeze()

            # 3. 筛选 Top-K (去除偏离语义的描述)
            k = max(1, int(len(texts) * top_k_ratio))
            # topk_vals, topk_indices
            _, idxs = torch.topk(sims, k)

            # 4. 聚合选出来的特征
            selected_feats = feats[idxs]
            final_proto = selected_feats.mean(dim=0)  # [512]
        else:
            # 如果只有一条描述，直接用
            final_proto = feats.squeeze()

        # 再次归一化
        final_proto = F.normalize(final_proto, dim=0)
        class_prototypes.append(final_proto)

    # 5. 堆叠并保存
    # Shape: [Num_Classes, 512]
    all_embeddings = torch.stack(class_prototypes, dim=0)

    print(f">> 保存生成的文本原型至: {save_path}")
    torch.save(all_embeddings, save_path)

    return all_embeddings





import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from clip import clip
from tqdm import tqdm
import numpy as np

# === 导入自定义模块 ===
from dataset.Adataset import AgDataset
from dataset.config import PVi_PDc_common_classes
from network import DAModel, grad_reverse
from utils.loss import PrototypeLoss, TopologyLoss, DecouplingLoss, EvidenceLoss
from load_text import generate_gpt_embeddings
from utils.utils import get_logger, set_seed, test_ours, test_per_class


class Args:
    root_dir = '/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/Agriculture'
    source_domain = 'PlantVillage'
    target_domain = 'PlantDoc'      # PlantVillage, PlantDoc, Plant Pathology, AI Challenger, PlantDiseases
    dataset_name = 'PDc-PDs'  # 用于聚类缓存文件的命名，不影响输出目录

    json_path = './dataset/description.json'
    clip_path = './clip/ViT-B-16.pt'

    # 基础输出目录
    base_output_dir = './train_output/ours'

    batch_size = 64
    lr = 1e-4          # 1e-4
    epochs = 200
    top_k = 0.4
    device = 'cuda:0'

    shared_classes = sorted(list(PVi_PDc_common_classes))
    num_classes = len(shared_classes)


def train():
    device = Args.device

    # === 1. 准备输出目录和日志 ===
    # 格式: ./output/PlantVillage2PlantDoc/
    task_name = f"{Args.source_domain}2{Args.target_domain}"
    output_dir = os.path.join(Args.base_output_dir, task_name)

    logger = get_logger(output_dir)

    logger(f"=== Training Start ===")
    logger(f"Device: {device}")
    logger(f"Task: {task_name}")
    logger(f"Output Dir: {output_dir}")
    logger(f"Classes: {Args.num_classes} | Epochs: {Args.epochs} | Batch: {Args.batch_size}")
    logger("-" * 50)

    # === 2. 准备数据 ===
    logger(">>> Loading Datasets...")
    source_ds = AgDataset(Args.root_dir, Args.source_domain, Args.shared_classes, is_source=True, phase='train', ratio=0.2)
    target_ds = AgDataset(Args.root_dir, Args.target_domain, Args.shared_classes, is_source=False, phase='train')
    test_ds = AgDataset(Args.root_dir, Args.target_domain, Args.shared_classes, is_source=False, phase='test')

    source_loader = DataLoader(source_ds, batch_size=Args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    target_loader = DataLoader(target_ds, batch_size=Args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=Args.batch_size, shuffle=False, num_workers=4)

    logger(f"Source Samples: {len(source_ds)} | Batches: {len(source_loader)}")
    logger(f"Target Samples: {len(target_ds)} | Batches: {len(target_loader)}")

    # === 3. 准备文本特征 ===
    logger(">>> Generating Text Attributes...")
    text_embeddings = generate_gpt_embeddings(
        dataset_name=Args.dataset_name,
        class_names=Args.shared_classes,
        json_file=Args.json_path,
        clip_weight_path=Args.clip_path,
        device=device,
        top_k_ratio=Args.top_k
    ).detach()

    # 消融gpt生成文本
    # template = "A plant pathology photo of {}"
    # clean_class_name = [name.replace('_', ' ') for name in Args.shared_classes]
    # prompts = [template.format(name) for name in clean_class_name]
    # text_input = clip.tokenize(prompts).to(device)
    # with torch.no_grad():
    #     clip_model, _ = clip.load(Args.clip_path, device=device, jit=False)
    #     text_feature = clip_model.encode_text(text_input).to(device)
    #     text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    # text_embeddings = text_feature.float().detach()
    # del clip_model
    # torch.cuda.empty_cache()
    # alpha = 0.3
    # text_embeddings = alpha * gpt_embeddings + (1-alpha) * temp_embeddings

    # === 4. 初始化模型 ===
    logger(">>> Initializing DAModel...")
    model = DAModel(Args.num_classes, Args.clip_path, device).to(device)
    params = [
        # {"params": model.clip_model.visual.parameters(), "lr": Args.lr * 0.1},
        {"params": model.adapter.parameters(), "lr": Args.lr},
        {"params": model.discriminator.parameters(), "lr": Args.lr},
        {"params": model.prototypes, "lr": Args.lr},  # 原型可以更新快一点 * 10
        {"params": model.spatial_bridge.parameters(), "lr": Args.lr},
        {"params": model.cross_attn.parameters(), "lr": Args.lr}
    ]

    optimizer = optim.AdamW(params, lr=Args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Args.epochs)

    # model.init_prototypes_with_source(source_loader, text_embeddings, device)
    model.prototypes.data = text_embeddings.clone()
    # model.prototypes.requires_grad = False

    # class_weights = torch.tensor([5.0, 5.0, 0.5]).to(device)      # PP dataset
    # criterion_ce = nn.CrossEntropyLoss(weight=class_weights).to(device)
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_edl = EvidenceLoss(Args.num_classes).to(device)
    criterion_bce = nn.BCEWithLogitsLoss().to(device)
    criterion_proto = PrototypeLoss().to(device)
    criterion_decouple = DecouplingLoss().to(device)
    criterion_topo = TopologyLoss().to(device)

    # === 5. 训练循环 ===
    best_acc = 0.3

    for epoch in range(Args.epochs):
        model.train()

        source_correct = 0
        source_total = 0

        epoch_loss = 0.0
        epoch_loss_cls = 0.0
        epoch_loss_proto = 0.0
        epoch_loss_dc = 0.0
        epoch_loss_domain = 0.0
        epoch_loss_topo = 0.0
        epoch_loss_pseudo = 0.0

        # === 迭代器逻辑修正：选择最长的数据集作为 Epoch 长度 ===
        steps_per_epoch = max(len(source_loader), len(target_loader))

        iter_s = iter(source_loader)
        iter_t = iter(target_loader)

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{Args.epochs}")

        for i in pbar:
            # --- 读取源域数据 ---
            try:
                s_imgs, s_labels, _ = next(iter_s)
            except StopIteration:
                iter_s = iter(source_loader)
                s_imgs, s_labels, _ = next(iter_s)

            # --- 读取目标域数据 ---
            try:
                t_imgs, _, _ = next(iter_t)
            except StopIteration:
                iter_t = iter(target_loader)
                t_imgs, _, _ = next(iter_t)

            s_imgs, s_labels = s_imgs.to(device), s_labels.to(device)
            t_imgs = t_imgs.to(device)

            # --- Forward ---
            out_s = model(s_imgs, text_embeddings)
            out_t = model(t_imgs, text_embeddings)

            # --- Losses ---
            with torch.no_grad():
                _, s_pred = torch.max(out_s['logits'], 1)
                source_correct += (s_pred == s_labels).sum().item()
                source_total += s_labels.size(0)

            loss_cls = criterion_ce(out_s['logits'], s_labels)

            # 目标域伪标签损失 (Refined Pseudo-Labeling)
            logit_t = out_t['logits']
            T = 1.0
            probs_t = torch.softmax(logit_t / T, dim=1)
            max_prob, pseudo_label_t = torch.max(probs_t, dim=1)        # prob mean: 0.629 | prob max: 0.999
            # 获取不确定性
            unc_t = out_t['uncertainty'].squeeze()  # [B]   unc mean: 0.256 | unc max: 0.375

            mask = torch.zeros_like(max_prob, dtype=torch.bool)
            if epoch > 5:
                K = 2   # batch=64, class=27, k=2/3/4/5
                for c in range(Args.num_classes):
                    class_idx = (pseudo_label_t == c).nonzero(as_tuple=True)[0]
                    if len(class_idx) == 0:
                        continue
                    probs_c = max_prob[class_idx]
                    unc_c = unc_t[class_idx]
                    threshold = min(0.9, max(0.7, probs_c.mean().item()))
                    basic_mask_c = (probs_c.ge(threshold)) & (unc_c.le(0.3))     # 非常自信 (>0.9) 且 不确定性较低 (<0.3)
                    valid_idx = class_idx[basic_mask_c]
                    valid_probs = probs_c[basic_mask_c]
                    if len(valid_idx) == 0:
                        if len(class_idx) > 0:
                            max_val, max_pos = torch.max(probs_c, dim=0)
                            if max_val > 0.5:
                                final_idx = class_idx[max_pos].unsqueeze(0)
                                mask[final_idx] = True
                        continue
                    if len(valid_idx) > K:      # Top-K 截断 (防止单类霸榜)
                        _, top_k_pos = torch.topk(valid_probs, K)
                        final_idx = valid_idx[top_k_pos]    # 如果及格的太多，只取前 K 个分最高的
                    else:
                        final_idx = valid_idx
                    mask[final_idx] = True
            else:       # 如果是预热阶段 (epoch <= 5)，mask 保持全 False，不产生 Loss
                pass

            if mask.sum() > 0:      # select 4 sample in this batch.
                # 对伪标签样本使用 EDL Loss.
                loss_pseudo = criterion_edl(logit_t[mask], pseudo_label_t[mask], epoch)
                # 计算目标域特征与原型
                feat_t_select = out_t['f_weighted'][mask]
                label_t_select = pseudo_label_t[mask]
                loss_proto_t = criterion_proto(feat_t_select, model.prototypes, label_t_select)
            else:
                loss_pseudo = torch.tensor(0.0).to(device)
                loss_proto_t = torch.tensor(0.0).to(device)

            loss_proto_s = criterion_proto(out_s['f_weighted'], model.prototypes, s_labels)
            loss_proto = loss_proto_s + 0.5 * loss_proto_t

            loss_dc = criterion_decouple(out_s['origin_visual'], out_s['v_inv'], out_s['v_spec']) + \
                      criterion_decouple(out_t['origin_visual'], out_t['v_inv'], out_t['v_spec'])

            label_s = torch.zeros(s_imgs.size(0), 1).to(device)
            label_t = torch.ones(t_imgs.size(0), 1).to(device)
            loss_domain = criterion_bce(out_s['disc_pred'], label_s) + criterion_bce(out_t['disc_pred'], label_t)

            loss_topo = (criterion_topo(out_s['m_vis'], out_s['m_txt']) +
                         0.5 * criterion_topo(out_t['m_vis'], out_t['m_txt']))

            # 总损失
            w = 0.1 if epoch > 6 else 0.0
            loss = loss_cls + 0.5 * loss_proto + 0.01 * loss_pseudo + w * loss_dc + w * loss_domain + 0.5 * loss_topo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加 Loss
            epoch_loss += loss.item()
            epoch_loss_cls += loss_cls.item()
            epoch_loss_proto += loss_proto.item()
            epoch_loss_dc += loss_dc.item()
            epoch_loss_domain += loss_domain.item()
            epoch_loss_topo += loss_topo.item()
            epoch_loss_pseudo += loss_pseudo.item()

            pbar.set_postfix({'Loss': f"{loss.item():.2f}", 'Cls': f"{loss_cls.item():.2f}",
                              'pseudo': f"{loss_pseudo.item():.2f}"})
        scheduler.step()

        # === 记录 Epoch 日志 ===
        avg_loss = epoch_loss / steps_per_epoch
        avg_loss_cls = epoch_loss_cls / steps_per_epoch
        avg_loss_proto = epoch_loss_proto / steps_per_epoch
        avg_loss_dc = epoch_loss_dc / steps_per_epoch
        avg_loss_domain = epoch_loss_domain / steps_per_epoch
        avg_loss_topo = epoch_loss_topo / steps_per_epoch
        avg_loss_pseudo = epoch_loss_pseudo / steps_per_epoch

        source_acc = source_correct / source_total

        # 格式化详细日志字符串
        log_msg = (
            f"[Epoch {epoch + 1}] "
            f"Total: {avg_loss:.4f} | "
            f"Cls: {avg_loss_cls:.4f} | "
            f"Pseudo: {avg_loss_pseudo:.4f} | "  # 伪标签
            f"Adv: {avg_loss_domain:.4f} | "
            f"Proto: {avg_loss_proto:.4f} | "  # 原型
            f"DC: {avg_loss_dc:.4f} | "  # 解耦
            f"Topo: {avg_loss_topo:.4f} |"  # 拓扑
            f"S-acc: {source_acc:.4f}"
        )
        # === Test (每10轮 或 最后一轮) ===
        if (epoch + 1) % 5 == 0 or (epoch + 1) == Args.epochs:
            test_acc = test_ours(model, test_loader, text_embeddings, device,
                                 class_names=Args.shared_classes, logger=logger)
            # test_acc = test_per_class(model, test_loader, device, class_names=Args.shared_classes,
            #                           logger=logger, text_embeddings=text_embeddings)
            log_msg += f" | Test Acc: {test_acc:.4f}"

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                save_path = os.path.join(output_dir, 'best_model.pth')
                torch.save(model.state_dict(), save_path)
                log_msg += f" (Best! Saved to {save_path})"

        logger(log_msg)
    logger(f"=== Training Finished. Best Accuracy: {best_acc:.4f} ===")


if __name__ == '__main__':
    set_seed(seed=42)
    train()


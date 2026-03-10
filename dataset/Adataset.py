import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class AgDataset(Dataset):
    def __init__(self, root_dir, domain_name, common_classes, is_source=True, phase='train', img_size=224, ratio=1.0):
        self.root_path = os.path.join(root_dir, domain_name)
        self.is_source = is_source
        self.phase = phase
        self.img_size = img_size
        self.ratio = ratio

        # 保证类别ID固定
        self.classes = common_classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.transform = self._get_transforms()
        self.samples = []
        self._load_data()

        role = "Source" if self.is_source else "Target"
        print(f"[{role} | {phase.upper()}] Loaded {len(self.samples)} samples from {self.root_path}")

    def _get_transforms(self):
        # CLIP 官方归一化参数
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

        if self.phase == 'train':
            return transforms.Compose([
                transforms.Resize(self.img_size + 32),
                transforms.RandomResizedCrop(self.img_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
                transforms.Normalize(mean, std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def _load_data(self):
        domain_label = 0 if self.is_source else 1
        for cls_name in self.classes:
            cls_folder = os.path.join(self.root_path, cls_name)
            if not os.path.exists(cls_folder):
                continue

            images = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if self.ratio < 1.0 and self.is_source:
                images.sort()
                random.seed(42)
                random.shuffle(images)
                retain = max(1, int(len(images) * self.ratio))
                images = images[:retain]


            real_label = self.class_to_idx[cls_name]

            # 目标域训练数据不带标签 (设为-1)
            final_label = -1 if (not self.is_source and self.phase == 'train') else real_label

            for img_name in images:
                self.samples.append((os.path.join(cls_folder, img_name), final_label, domain_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, domain = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label, domain


# Resnet backbone的数据读取方式
class AgDataset_res(Dataset):
    def __init__(self, root_dir, domain_name, common_classes, is_source=True, phase='train', img_size=224, ratio=1.0):
        self.root_path = os.path.join(root_dir, domain_name)
        self.is_source = is_source
        self.phase = phase
        self.img_size = img_size
        self.ratio = ratio

        # 保证类别ID固定
        self.classes = common_classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.transform = self._get_transforms()
        self.samples = []
        self._load_data()

        role = "Source" if self.is_source else "Target"
        print(f"[{role} | {phase.upper()}] Loaded {len(self.samples)} samples from {self.root_path}")

    def _get_transforms(self):
        # resnet 官方归一化参数
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.phase == 'train':
            return transforms.Compose([
                transforms.Resize(self.img_size + 32),
                transforms.RandomResizedCrop(self.img_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def _load_data(self):
        domain_label = 0 if self.is_source else 1
        for cls_name in self.classes:
            cls_folder = os.path.join(self.root_path, cls_name)
            if not os.path.exists(cls_folder):
                continue

            images = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if self.ratio < 1.0 and self.is_source:
                images.sort()
                random.seed(42)
                random.shuffle(images)
                retain = max(1, int(len(images) * self.ratio))
                images = images[:retain]

            real_label = self.class_to_idx[cls_name]

            # 目标域训练数据不带标签 (设为-1)
            final_label = -1 if (not self.is_source and self.phase == 'train') else real_label

            for img_name in images:
                self.samples.append((os.path.join(cls_folder, img_name), final_label, domain_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, domain = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label, domain


class AgDataset_strong(Dataset):
    def __init__(self, root_dir, domain_name, common_classes, is_source=True, phase='train', img_size=224, ratio=1.0):
        self.root_path = os.path.join(root_dir, domain_name)
        self.is_source = is_source
        self.phase = phase
        self.img_size = img_size
        self.ratio = ratio

        # 保证类别ID固定
        self.classes = common_classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 初始化变换
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if self.phase == 'train':
            self.transform_weak = self._get_weak_transforms()
            self.transform_strong = self._get_strong_transforms()
        else:
            self.transform_test = self._get_test_transforms()

        self.samples = []
        self._load_data()

        role = "Source" if self.is_source else "Target"
        print(f"[{role} | {phase.upper()}] Loaded {len(self.samples)} samples from {self.root_path}")

    def _get_weak_transforms(self):
        """弱增强：标准的 Resize, Crop, Flip"""
        return transforms.Compose([
            transforms.Resize((self.img_size + 32, self.img_size + 32)),
            transforms.RandomCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def _get_strong_transforms(self):
        """强增强：RandAugment 风格 (颜色抖动, 灰度, 高斯模糊)"""
        return transforms.Compose([
            transforms.Resize((self.img_size + 32, self.img_size + 32)),
            transforms.RandomCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            # 强颜色抖动
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # 随机灰度
            transforms.RandomGrayscale(p=0.2),
            # 高斯模糊
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23)  # kernel size 需为奇数
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def _get_test_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def _load_data(self):
        domain_label = 0 if self.is_source else 1
        for cls_name in self.classes:
            cls_folder = os.path.join(self.root_path, cls_name)
            if not os.path.exists(cls_folder):
                continue

            images = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Source 域数据抽样
            if self.ratio < 1.0 and self.is_source:
                images.sort()
                random.seed(42)
                random.shuffle(images)
                retain = max(1, int(len(images) * self.ratio))
                images = images[:retain]

            real_label = self.class_to_idx[cls_name]
            # 目标域训练数据不带标签 (设为-1)
            final_label = -1 if (not self.is_source and self.phase == 'train') else real_label

            for img_name in images:
                self.samples.append((os.path.join(cls_folder, img_name), final_label, domain_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, domain = self.samples[idx]
        img = Image.open(path).convert('RGB')

        if self.phase == 'train':
            # 训练阶段：同时返回弱增强和强增强视图
            img_weak = self.transform_weak(img)
            img_strong = self.transform_strong(img)
            return img_weak, img_strong, label, domain
        else:
            # 测试阶段：只返回标准视图
            img = self.transform_test(img)
            return img, label, domain


# DG-PLDR论文的数据读取方式
class AgDataset_DG(Dataset):
    def __init__(self, root_dir, domain_name, common_classes, phase='train', img_size=224, ratio=1.0):
        self.root_path = os.path.join(root_dir, domain_name)
        self.phase = phase
        self.img_size = img_size
        self.classes = common_classes
        self.ratio = ratio
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        self._load_data()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # === 训练 transforms: 仅做 Resize/Crop 和 Tensor化 ===
        self.resize_crop = transforms.Compose([
            transforms.Resize(self.img_size + 32),
            transforms.RandomResizedCrop(self.img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])

        # === 样式增强: 颜色扰动 + 高斯模糊 (论文所述) ===
        self.style_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        # === 测试 transforms ===
        self.test_transform = transforms.Compose([
            transforms.Resize(self.img_size + 32),
            transforms.RandomResizedCrop(self.img_size, scale=(0.2, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        print(f"[{phase.upper()}] Loaded {len(self.samples)} samples from {self.root_path}")

    def _load_data(self):
        for cls_name in self.classes:
            cls_folder = os.path.join(self.root_path, cls_name)
            if not os.path.exists(cls_folder): continue
            images = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

            if self.ratio < 1.0 and self.phase == 'train':
                images.sort()
                random.seed(42)
                random.shuffle(images)
                retain = max(1, int(len(images) * self.ratio))
                images = images[:retain]

            label = self.class_to_idx[cls_name]
            for img in images:
                self.samples.append((os.path.join(cls_folder, img), label))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')

        if self.phase == 'train':
            # 1. 基础几何变换 (保证两张图空间对齐)
            img_base = self.resize_crop(img)

            # 2. 原图输入
            img_orig = self.normalize(img_base)

            # 3. 样式增强输入
            # 对同一张 base 图做样式增强
            img_aug = self.style_aug(img_base)
            img_aug = self.normalize(img_aug)

            return img_orig, img_aug, label
        else:
            return self.test_transform(img), label


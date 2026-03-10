import os
import json
import shutil
from tqdm import tqdm

# --- AI Challenger 2018 农作物病害 61类 完整映射表 ---
# 格式: ID: "作物_病害_程度" (中文对应在注释中)
ID_TO_NAME = {
    "0": "Apple_Healthy",  # 苹果-健康
    "1": "Apple_Scab_General",  # 苹果-黑星病-一般
    "2": "Apple_Scab_Severe",  # 苹果-黑星病-严重
    "3": "Apple_Frogeye",  # 苹果-灰斑病(蛙眼叶斑)
    "4": "Apple_Cedar_Rust_General",  # 苹果-雪松锈病-一般
    "5": "Apple_Cedar_Rust_Severe",  # 苹果-雪松锈病-严重
    "6": "Cherry_Healthy",  # 樱桃-健康
    "7": "Cherry_Powdery_Mildew_General",  # 樱桃-白粉病-一般
    "8": "Cherry_Powdery_Mildew_Severe",  # 樱桃-白粉病-严重
    "9": "Corn_Healthy",  # 玉米-健康
    "10": "Corn_Gray_Leaf_Spot_General",  # 玉米-灰斑病-一般
    "11": "Corn_Gray_Leaf_Spot_Severe",  # 玉米-灰斑病-严重
    "12": "Corn_Common_Rust_General",  # 玉米-锈病-一般
    "13": "Corn_Common_Rust_Severe",  # 玉米-锈病-严重
    "14": "Corn_Northern_Leaf_Blight_General",  # 玉米-大斑病-一般
    "15": "Corn_Northern_Leaf_Blight_Severe",  # 玉米-大斑病-严重
    "16": "Corn_dwarf_mosaic_virus",  # 玉米-花叶病毒
    "17": "Grape_Healthy",  # 葡萄-健康
    "18": "Grape_Black_Rot_General",  # 葡萄-黑腐病-一般
    "19": "Grape_Black_Rot_Severe",  # 葡萄-黑腐病-严重
    "20": "Grape_Black_Measles_General",  # 葡萄-褐斑病(轮斑病)-一般
    "21": "Grape_Black_Measles_Severe",  # 葡萄-褐斑病(轮斑病)-严重
    "22": "Grape_Leaf_Blight_General",  # 葡萄-叶斑病-一般
    "23": "Grape_Leaf_Blight_Severe",  # 葡萄-叶斑病-严重
    "24": "Citrus_Healthy",  # 柑橘-健康
    "25": "Citrus_Greening_General",  # 柑橘-黄龙病-一般
    "26": "Citrus_Greening_Severe",  # 柑橘-黄龙病-严重
    "27": "Peach_Healthy",  # 桃-健康
    "28": "Peach_Bacterial_Spot_General",  # 桃-细菌性穿孔病-一般
    "29": "Peach_Bacterial_Spot_Severe",  # 桃-细菌性穿孔病-严重
    "30": "Pepper_Healthy",  # 辣椒-健康
    "31": "Pepper_Bacterial_Spot_General",  # 辣椒-细菌性斑点病-一般
    "32": "Pepper_Bacterial_Spot_Severe",  # 辣椒-细菌性斑点病-严重
    "33": "Potato_Healthy",  # 马铃薯-健康
    "34": "Potato_Early_Blight_General",  # 马铃薯-早疫病-一般
    "35": "Potato_Early_Blight_Severe",  # 马铃薯-早疫病-严重
    "36": "Potato_Late_Blight_General",  # 马铃薯-晚疫病-一般
    "37": "Potato_Late_Blight_Severe",  # 马铃薯-晚疫病-严重
    "38": "Strawberry_Healthy",  # 草莓-健康
    "39": "Strawberry_Leaf_Scorch_General",  # 草莓-叶枯病-一般
    "40": "Strawberry_Leaf_Scorch_Severe",  # 草莓-叶枯病-严重
    "41": "Tomato_Healthy",  # 番茄-健康
    "42": "Tomato_Powdery_Mildew_General",  # 番茄-白粉病-一般
    "43": "Tomato_Powdery_Mildew_Severe",
    "44": "Tomato_Bacterial_Spot_General",  # 番茄-细菌性斑点病-一般
    "45": "Tomato_Bacterial_Spot_Severe",  # 番茄-细菌性斑点病-严重
    "46": "Tomato_Early_Blight_General",  # 番茄-早疫病-一般
    "47": "Tomato_Early_Blight_Severe",  # 番茄-早疫病-严重
    "48": "Tomato_Late_Blight_General",  # 番茄-晚疫病-一般
    "49": "Tomato_Late_Blight_Severe",  # 番茄-晚疫病-严重
    "50": "Tomato_Leaf_Mold_General",  # 番茄-叶霉病-一般
    "51": "Tomato_Leaf_Mold_Severe",  # 番茄-叶霉病-严重
    "52": "Tomato_Target_Spot_General",  # 番茄-靶斑病-一般
    "53": "Tomato_Target_Spot_Severe",  # 番茄-靶斑病-严重
    "54": "Tomato_Septoria_Leaf_Spot_General",  # 番茄-斑枯病-一般
    "55": "Tomato_Septoria_Leaf_Spot_Severe",  # 番茄-斑枯病-严重
    "56": "Tomato_Spider_Mites_General",  # 番茄-红蜘蛛损伤-一般
    "57": "Tomato_Spider_Mites_Severe",  # 番茄-红蜘蛛损伤-严重
    "58": "Tomato_Yellow_Leaf_Curl_Virus_General",  # 番茄-黄化曲叶病毒病-一般
    "59": "Tomato_Yellow_Leaf_Curl_Virus_Severe",  # 番茄-黄化曲叶病毒病-严重
    "60": "Tomato_Mosaic_Virus",  # 番茄-花叶病毒病
}


def organize_priority(image_src_dir, json_path, output_dir, mode='copy'):
    """
    优先移动原名文件，找不到则尝试移动副本文件，并统一重命名为标准格式。
    """
    print(f"读取 JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 统计计数器
    count_original = 0
    count_copy_fixed = 0
    count_missing = 0

    print("开始处理...")

    for item in tqdm(annotations):
        try:
            # 1. 获取基本信息
            target_name = item['image_id']  # 例如: abc.jpg
            class_id = str(item['disease_class'])
            folder_name = ID_TO_NAME.get(class_id, f"Class_{class_id}")

            # 准备路径
            target_folder = os.path.join(output_dir, folder_name)
            dst_path = os.path.join(target_folder, target_name)  # 目标永远是干净的 abc.jpg

            # --- 核心逻辑开始 ---

            # 路径 A: 原始文件名路径
            path_original = os.path.join(image_src_dir, target_name)

            # 路径 B: 副本文件名路径 (自动构建: abc.jpg -> abc - 副本.jpg)
            name_part, ext_part = os.path.splitext(target_name)
            name_copy = f"{name_part} - 副本{ext_part}"
            path_copy = os.path.join(image_src_dir, name_copy)

            final_src_path = None

            # 优先级判断
            if os.path.exists(path_original):
                final_src_path = path_original
                count_original += 1
            elif os.path.exists(path_copy):
                final_src_path = path_copy
                count_copy_fixed += 1
            else:
                count_missing += 1
                continue  # 两个都找不到，跳过

            # --- 执行移动/复制 ---

            # 确保目标文件夹存在
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # 执行操作
            if mode == 'move':
                shutil.move(final_src_path, dst_path)
            else:
                shutil.copy(final_src_path, dst_path)

        except Exception as e:
            print(f"Error processing {item}: {e}")
            count_missing += 1

    print("\n" + "=" * 30)
    print("处理完成！统计报告：")
    print(f"1. 原名匹配成功: {count_original} 张")
    print(f"2. 副本修复成功: {count_copy_fixed} 张 (已自动改回原名)")
    print(f"3. 依然缺失:     {count_missing} 张")
    print("-" * 20)
    print(f"总计找到: {count_original + count_copy_fixed} / {len(annotations)}")
    print(f"输出目录: {output_dir}")
    print("=" * 30)


def organize_from_txt(txt_path, image_src_dir, output_dir, mode='copy'):
    print(f"读取列表文件: {txt_path}")

    # 1. 读取 txt 所有行
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"找到 {len(lines)} 条记录，准备开始整理...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success_count = 0
    fail_count = 0

    for line in tqdm(lines):
        line = line.strip()
        if not line: continue

        # 2. 解析每一行
        # txt格式: "路径/文件名ID" (空格) "类别ID"
        # 例如: AgriculturalDisease_trainingset/images\1_0.jpg 1
        parts = line.split()
        if len(parts) < 2:
            continue

        full_path_str = parts[0]
        class_id = parts[1]

        # 3. 提取纯文件名 (处理可能出现的 / 或 \ 符号)
        # 无论前面路径写得多么乱，我们只取最后的文件名 "1_0.jpg"
        filename = os.path.basename(full_path_str.replace('\\', '/'))

        # 4. 获取目标文件夹名
        folder_name = ID_TO_NAME.get(class_id, f"Class_{class_id}")
        target_folder = os.path.join(output_dir, folder_name)

        # 5. 寻找源文件 (加入副本容错逻辑)
        src_path_standard = os.path.join(image_src_dir, filename)

        # 构建副本文件名: 1_0.jpg -> 1_0 - 副本.jpg
        name_part, ext_part = os.path.splitext(filename)
        src_path_copy = os.path.join(image_src_dir, f"{name_part} - 副本{ext_part}")

        final_src_path = None

        if os.path.exists(src_path_standard):
            final_src_path = src_path_standard
        elif os.path.exists(src_path_copy):
            final_src_path = src_path_copy
        else:
            # 两个都找不到
            # print(f"Missing: {filename}")
            fail_count += 1
            continue

        # 6. 执行移动/复制
        try:
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # 目标路径始终使用标准文件名 (去掉副本后缀)
            dst_path = os.path.join(target_folder, filename)

            if mode == 'move':
                shutil.move(final_src_path, dst_path)
            else:
                shutil.copy(final_src_path, dst_path)

            success_count += 1

        except Exception as e:
            print(f"Error: {e}")
            fail_count += 1

    print("\n" + "=" * 30)
    print("TXT 列表处理完成！")
    print(f"成功整理: {success_count} 张")
    print(f"失败/缺失: {fail_count} 张")
    print(f"输出目录: {output_dir}")
    print("=" * 30)



def merge_severity_folders(dataset_root):
    """
    遍历数据集目录，将 _General 和 _Severe 后缀的文件夹合并到不带后缀的病害文件夹中。
    """
    if not os.path.exists(dataset_root):
        print(f"错误: 找不到目录 {dataset_root}")
        return

    # 获取所有子文件夹
    all_folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]

    print(f"检测到 {len(all_folders)} 个子文件夹，准备合并...")

    merged_count = 0
    moved_images_count = 0

    # 遍历每一个文件夹
    for folder_name in tqdm(all_folders):
        old_folder_path = os.path.join(dataset_root, folder_name)

        # 1. 判断是否需要合并
        target_folder_name = None

        if folder_name.endswith("_General"):
            # 去掉 "_General" (8个字符)
            target_folder_name = folder_name[:-8]
        elif folder_name.endswith("_Severe"):
            # 去掉 "_Severe" (7个字符)
            target_folder_name = folder_name[:-7]

        # 如果不是这就两种结尾（比如是 Apple_Healthy），则跳过
        if target_folder_name is None:
            continue

        # 2. 创建目标文件夹 (例如 Apple_Scab)
        target_folder_path = os.path.join(dataset_root, target_folder_name)
        if not os.path.exists(target_folder_path):
            os.makedirs(target_folder_path)

        # 3. 移动图片
        images = os.listdir(old_folder_path)
        for img_name in images:
            src_img = os.path.join(old_folder_path, img_name)
            dst_img = os.path.join(target_folder_path, img_name)

            # 防止重名覆盖（虽然概率极低，但为了安全）
            if os.path.exists(dst_img):
                name, ext = os.path.splitext(img_name)
                # 如果重名，加个 _dup 后缀
                dst_img = os.path.join(target_folder_path, f"{name}_dup{ext}")

            try:
                shutil.move(src_img, dst_img)
                moved_images_count += 1
            except Exception as e:
                print(f"移动失败: {src_img} -> {e}")

        # 4. 删除空的旧文件夹
        try:
            os.rmdir(old_folder_path)
            merged_count += 1
        except OSError:
            print(f"警告: 文件夹 {folder_name} 不为空，未删除。")

    print("\n" + "=" * 30)
    print("合并完成！")
    print(f"共合并了: {merged_count} 个文件夹")
    print(f"共移动了: {moved_images_count} 张图片")

    # 统计一下现在的文件夹数量
    current_folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    print(f"当前剩余类别数: {len(current_folders)}")
    print("=" * 30)


if __name__ == '__main__':
    # 路径配置
    source_images = 'D:/Anaconda/code/0dataset/Agriculture/AI Challenger/AgriculturalDisease_validationset/images'
    json_file = 'D:/Anaconda/code/0dataset/Agriculture/AI Challenger/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json'

    # 新发现的 txt 文件路径 (请修改为你电脑上的实际位置)
    txt_list_file = 'D:/Anaconda/code/0dataset/Agriculture/AI Challenger/AgriculturalDisease_validationset/ttest_list.txt'

    target_output = 'D:/Anaconda/code/0dataset/Agriculture/AI Challenger/AI Challenger 2018'

    # 建议先用 'copy' 模式，确认无误后再用 'move'
    # organize_priority(source_images, json_file, target_output, mode='copy')

    # organize_from_txt(txt_list_file, source_images, target_output, mode='copy')

    merge_severity_folders(target_output)


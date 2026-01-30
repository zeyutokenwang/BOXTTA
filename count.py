import os
import csv
from pathlib import Path

# ================= 配置区域 =================
# 填入你保存切片的根目录
BASE_DIR = '/home/token/SAM-TTA/data/BraTS_SSA_t2f_2D'
OUTPUT_CSV = '/home/token/SAM-TTA/data/BraTS_SSA_t2f_2D/all.csv'

# 目录名称
IMAGE_FOLDER_NAME = 'image'
MASK_FOLDER_NAME = 'mask'
# ===========================================

def generate_csv():
    image_dir = os.path.join(BASE_DIR, IMAGE_FOLDER_NAME)
    mask_dir = os.path.join(BASE_DIR, MASK_FOLDER_NAME)

    if not os.path.exists(image_dir):
        print(f"错误: 找不到目录 {image_dir}")
        return

    # 获取 image 目录下所有的 nii.gz 文件
    # 使用 sorted 确保文件按名称排序，方便查看
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii')])
    
    pairs = []
    missing_masks = 0

    for filename in image_files:
        # 检查 mask 目录下是否存在同名文件
        mask_path = os.path.join(mask_dir, filename)
        
        if os.path.exists(mask_path):
            # 构造 CSV 中的路径格式
            # image/文件名, mask/文件名
            img_rel_path = f"{IMAGE_FOLDER_NAME}/{filename}"
            mask_rel_path = f"{MASK_FOLDER_NAME}/{filename}"
            pairs.append([img_rel_path, mask_rel_path])
        else:
            missing_masks += 1

    # 写入 CSV 文件
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['image', 'label'])
        # 写入数据
        writer.writerows(pairs)

    print(f"--- 处理完成 ---")
    print(f"成功匹配并写入: {len(pairs)} 对文件")
    if missing_masks > 0:
        print(f"警告: 有 {missing_masks} 个 image 文件没有找到对应的 mask")
    print(f"CSV 文件保存至: {OUTPUT_CSV}")

    # 简单统计病例数（使用之前的逻辑验证）
    import re
    patient_ids = set(re.findall(r'BraTS-SSA-\d{5}', "".join([p[0] for p in pairs])))
    print(f"包含的唯一病例总数: {len(patient_ids)}")

if __name__ == "__main__":
    generate_csv()
import os
import nibabel as nib
import numpy as np
import logging
import json
from pathlib import Path

# ================= 配置区域 =================
# 路径设置
SOURCE_DIR = '/mnt/d/datasets/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2'
TARGET_IMAGE_DIR = '/home/token/SAM-TTA/data/BraTS_SSA_t2f_2D/image'  # 存放 t2w 切片
TARGET_MASK_DIR = '/home/token/SAM-TTA/data/BraTS_SSA_t2f_2D/mask'    # 存放 TC mask 切片

# 筛选阈值
MIN_TUMOR_AREA = 1000  # 只有 TC 像素点超过这个值的切片才会被保留 (可根据需要调整)

# 标签定义 (BraTS 2023)
# Label 1: Necrotic/Non-enhancing Tumor Core (NCR/NET)
# Label 2: Peritumoral Edema (ED)
# Label 3: Enhancing Tumor (ET)
# TC (Tumor Core) = Label 1 + Label 3
TC_LABELS = [1, 3] 

# ===========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs(TARGET_IMAGE_DIR, exist_ok=True)
os.makedirs(TARGET_MASK_DIR, exist_ok=True)

def find_t2f_file(seg_path):
    """根据seg文件路径找到对应的t2f文件路径"""
    return seg_path.replace('-seg.nii', '-t2f.nii')

def process_case(seg_path):
    """处理一个病例：读取seg和t2f，筛选并合并TC标签"""
    t2f_path = find_t2f_file(seg_path)
    
    if not os.path.exists(t2f_path):
        logger.warning(f"找不到对应的t2f文件: {t2f_path}")
        return 0
    
    try:
        # 加载数据
        seg_obj = nib.load(seg_path)
        t2f_obj = nib.load(t2f_path)
        
        seg_data = seg_obj.get_fdata()
        t2f_data = t2f_obj.get_fdata()
        
        patient_id = os.path.basename(seg_path).replace('-seg.nii', '')
        
        valid_slices = 0
        slices_count = seg_data.shape[2]
        
        for i in range(slices_count):
            seg_slice = seg_data[:, :, i]
            t2f_slice = t2f_data[:, :, i]
            
            # 1. 提取 TC 区域并合并标签 (1 和 3 变为 1, 其余变为 0)
            tc_mask = np.isin(seg_slice, TC_LABELS).astype(np.uint8)
            
            # 2. 面积阈值筛选
            tumor_pixel_count = np.sum(tc_mask)
            if tumor_pixel_count >= MIN_TUMOR_AREA:
                # 生成文件名
                slice_name = f"{patient_id}-t2f_{i:04d}.nii"
                
                # 保存 Mask (TC 合并后的二值图)
                mask_nii = nib.Nifti1Image(tc_mask, seg_obj.affine, seg_obj.header)
                nib.save(mask_nii, os.path.join(TARGET_MASK_DIR, slice_name))
                
                # 保存对应的 Image (t2f 原始强度图)
                img_nii = nib.Nifti1Image(t2f_slice, t2f_obj.affine, t2f_obj.header)
                nib.save(img_nii, os.path.join(TARGET_IMAGE_DIR, slice_name))
                
                valid_slices += 1
                
        return valid_slices

    except Exception as e:
        logger.error(f"处理病例 {seg_path} 出错: {str(e)}")
        return 0

def main():
    logger.info("开始提取 TC 区域切片...")
    
    # 搜索所有的 seg 文件
    seg_files = []
    for root, _, files in os.walk(SOURCE_DIR):
        for f in files:
            if f.endswith('-seg.nii'):
                seg_files.append(os.path.join(root, f))
    
    logger.info(f"找到 {len(seg_files)} 个病例")
    
    total_saved = 0
    used_cases = 0
    
    for seg_path in seg_files:
        count = process_case(seg_path)
        if count > 0:
            total_saved += count
            used_cases += 1
            logger.info(f"病例 {os.path.basename(seg_path)}: 保留了 {count} 张切片")
        else:
            logger.warning(f"病例 {os.path.basename(seg_path)}: 无符合条件的 TC 切片，已跳过")

    logger.info("="*50)
    logger.info(f"任务完成！")
    logger.info(f"总病例数: {len(seg_files)}")
    logger.info(f"实际使用的病例数: {used_cases}")
    logger.info(f"生成的 TC 切片总数: {total_saved}")
    logger.info(f"结果保存至: \n {TARGET_IMAGE_DIR} \n {TARGET_MASK_DIR}")

if __name__ == "__main__":
    main()
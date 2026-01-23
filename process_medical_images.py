import os
import nibabel as nib
import logging
import time
import json
from pathlib import Path

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/token/SAM-TTA/image_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义路径
SOURCE_DIR = '/mnt/d/datasets/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData'
TARGET_DIR = '/home/token/SAM-TTA/data/BraTS_PED_t2f_2D/mask'
PROGRESS_FILE = '/home/token/SAM-TTA/processing_progress.json'

# 确保目标目录存在
os.makedirs(TARGET_DIR, exist_ok=True)

# 加载处理进度
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
else:
    progress = {
        'processed_files': [],
        'total_files': 0,
        'success_count': 0,
        'failed_files': []
    }

def get_patient_id(filename):
    """从文件名中提取患者ID，如BraTS-PED-00108-000-t2f.nii.gz -> BraTS-PED-00108-000-t2f"""
    base_name = os.path.basename(filename)
    if base_name.endswith('.nii.gz'):
        return base_name[:-10] + "t2f"
    elif base_name.endswith('.nii'):
        return base_name[:-4]
    return base_name

def process_image(file_path):
    """处理单个医学影像文件，生成2D切片"""
    try:
        logger.info(f"开始处理文件: {file_path}")
        
        # 加载NIfTI文件
        img = nib.load(file_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        
        # 获取数据维度
        if data.ndim != 3:
            logger.error(f"文件 {file_path} 不是3D数据，跳过处理")
            return False, "不是3D数据"
        
        # 获取切片数量
        slices_count = data.shape[2]
        logger.info(f"文件 {file_path} 包含 {slices_count} 个切片")
        
        # 提取患者ID
        patient_id = get_patient_id(file_path)
        
        # 生成并保存2D切片
        for i in range(slices_count):
            # 提取单个切片
            slice_data = data[:, :, i]
            
            # 创建新的NIfTI文件
            slice_img = nib.Nifti1Image(slice_data, affine, header)
            
            # 生成切片文件名
            slice_filename = f"{patient_id}_{i:04d}.nii.gz"
            slice_path = os.path.join(TARGET_DIR, slice_filename)
            
            # 保存切片
            nib.save(slice_img, slice_path)
        
        logger.info(f"成功处理文件 {file_path}，生成 {slices_count} 个切片")
        return True, slices_count
        
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        return False, str(e)

def main():
    """主函数"""
    start_time = time.time()
    logger.info("开始医学影像切片处理任务")
    
    # 收集所有需要处理的文件
    all_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if ('seg' in file.lower()) and (file.endswith('.nii') or file.endswith('.nii.gz')):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    total_files = len(all_files)
    logger.info(f"共发现 {total_files} 个需要处理的seg文件")
    progress['total_files'] = total_files
    
    # 过滤已处理的文件
    pending_files = [f for f in all_files if f not in progress['processed_files']]
    logger.info(f"还有 {len(pending_files)} 个文件需要处理")
    
    # 处理剩余文件
    for file_path in pending_files:
        success, result = process_image(file_path)
        
        # 更新进度
        progress['processed_files'].append(file_path)
        
        if success:
            progress['success_count'] += result
        else:
            progress['failed_files'].append({
                'file_path': file_path,
                'error': result
            })
        
        # 保存进度
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=4)
    
    # 生成处理报告
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    report = {
        'total_files_checked': total_files,
        'total_files_processed': len(progress['processed_files']),
        'successfully_sliced': progress['success_count'],
        'failed_files_count': len(progress['failed_files']),
        'failed_files': progress['failed_files'],
        'total_time_seconds': elapsed_time,
        'total_time_formatted': time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    }
    
    # 保存报告
    report_path = '/home/token/SAM-TTA/processing_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # 打印报告
    logger.info("\n" + "="*50)
    logger.info("医学影像切片处理报告")
    logger.info("="*50)
    logger.info(f"检查的文件总数: {report['total_files_checked']}")
    logger.info(f"已处理的文件总数: {report['total_files_processed']}")
    logger.info(f"成功生成的切片数: {report['successfully_sliced']}")
    logger.info(f"处理失败的文件数: {report['failed_files_count']}")
    logger.info(f"总处理时间: {report['total_time_formatted']}")
    
    if report['failed_files_count'] > 0:
        logger.info("\n失败的文件列表:")
        for failed in report['failed_files']:
            logger.info(f"- {failed['file_path']}: {failed['error']}")
    
    logger.info("\n处理完成！")
    logger.info(f"报告已保存到: {report_path}")

if __name__ == "__main__":
    main()
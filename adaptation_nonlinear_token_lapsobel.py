import argparse
import importlib.util
import os
import yaml
import time
import lightning as L
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader
import random
from configs.config import cfg
from losses import DiceLoss, FocalLoss, ContraLoss
from datasets import call_load_dataset

from model import Model
from sam_lora import LoRA_Sam
from utils.eval_utils import AverageMeter, calc_iou, validate, get_prompts, validate_med, test_med, calculate_metrics
from utils.tools import copy_model, create_csv, check_grad, momentum_update, reduce_instances
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import torch.optim as optim
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from math import comb

from utils.lapsobel import LearnableBezierTransform
#from utils.nonlinear import LearnableBezierTransform

def expand_bbox(x, y, w, h, width, height, expand_ratio=0.25):
    new_w = w * (1 + expand_ratio)
    new_h = h * (1 + expand_ratio)
    
    new_x = max(0, x - (new_w - w) / 2)
    new_y = max(0, y - (new_h - h) / 2)
    
    new_x2 = min(width, x + w + (new_w - w) / 2)
    new_y2 = min(height, y + h + (new_h - h) / 2)
    
    return new_x, new_y, new_x2, new_y2

def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    anchor_model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    all_dataloader: DataLoader,
    num_iters: int,
):
    """The SAM training loop with Recursive Box Refinement."""
    data_time = AverageMeter()
    dice_loss = DiceLoss()
    end = time.time()
    num_epochs = 1
    
    # --- 原有记录变量完全保留 ---
    dice_scores = AverageMeter()
    assd_scores = AverageMeter() 
    hd95_scores = AverageMeter() 
    batch_time = AverageMeter()
    iou_losses = AverageMeter()
    dice_losses = AverageMeter()
    ent_losses = AverageMeter()
    total_losses = AverageMeter()
    case_results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform_auto = LearnableBezierTransform().to(device)
    optimizer_auto = optim.Adam(transform_auto.parameters(), lr=0.1)
    
    for epoch in range(0, num_epochs):
        for iter, data in enumerate(all_dataloader):
            loss = 0.
            data_time.update(time.time() - end)
            images_test, bboxes_test, bboxes_test_coarse, gt_masks_test, basenames, images_weak, images_strong, bboxes, gt_masks = data
            
            # 1. SBCT 转换
            transformed_images = transform_auto(images_test[0], basenames[0]).unsqueeze(0)
            batch_size = images_test.size(0)
            
            # --- 核心改进：闭环框校准逻辑 ---
            # A. 第一次 Pass：获取 Teacher 模型的所有输出（包括 anchor_res_masks）
            prompts_init = get_prompts(cfg, bboxes_test, gt_masks_test)
            anchor_image_embeds, first_masks, first_iou_predictions, first_res_masks = model(transformed_images, prompts_init)
            with torch.no_grad():
                # 确保这里解包 4 个变量：embeds, masks, ious, res_masks
                anchor_image_embeds, anchor_masks, anchor_iou_predictions, anchor_res_masks = anchor_model(transformed_images, prompts_init)
            
            # B. 使用 cv2.boundingRect 提取框并转换为 XYXY 格式
            refined_bboxes_list = []
            for i in range(batch_size):
                # 提取掩码并转为 2D NumPy
                mask_2d = (first_masks[i][0] > 0).cpu().numpy().astype(np.uint8).squeeze()
                iou_score = first_iou_predictions[i].mean().item()
                non_zero_pts = cv2.findNonZero(mask_2d)
                
                # 只有置信度高且真的有 Mask 时才校准
                if iou_score > 0.2 and non_zero_pts is not None:
                    # x, y 是左上角, w, h 是宽和高
                    x, y, w, h = cv2.boundingRect(non_zero_pts)
                    x1, y1, x2, y2 = expand_bbox(x, y, w, h, 1024, 1024, expand_ratio=0.0)
                    
                    # 构造 XYXY 格式的 Tensor
                    refined_box = torch.tensor([[x1, y1, x2, y2]], device=fabric.device, dtype=torch.float32)
                    refined_bboxes_list.append(refined_box)
                else:
                    # 如果校准失败，直接使用原始的 bboxes_test
                    refined_bboxes_list.append(bboxes_test[i].to(fabric.device).float())
            print("图像框信息")
            print(bboxes_test[0])
            print(refined_bboxes_list[0])
            # C. 第二次 Pass
            prompts_refined = get_prompts(cfg, refined_bboxes_list, gt_masks_test)

            # 执行 Student 模型的前向传播（带梯度）
            # 这里产生的变量名 pred_masks, pred_iou_predictions, pred_res_masks 将进入你下方的 zip 循环
            # pred_image_embeds, pred_masks, pred_iou_predictions, pred_res_masks = model(transformed_images, prompts_refined)
            
            
            num_masks = sum(len(pred_mask) for pred_mask in first_masks)
            loss_ent = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)

            for i, (pred_mask, anchor_mask, iou_prediction, anchor_res, pred_res) in enumerate(zip(first_masks, anchor_masks, first_iou_predictions, anchor_res_masks, first_res_masks)):
                iou_score = iou_prediction.mean()
                anchor_mask = (anchor_mask > 0.).float().detach() # 伪标签脱离梯度
                anchor_res = (anchor_res > 0.).float().detach()
                
                batch_iou = calc_iou(pred_mask, anchor_mask)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks
                loss_dice += dice_loss(pred_mask, anchor_mask)*(iou_score*iou_score*iou_score)
                loss_dice += dice_loss(pred_res, anchor_res)*(iou_score*iou_score*iou_score)
                loss_iou += (1.0 - iou_prediction.mean())
                
            loss_total = loss_iou + loss_dice*0.5

            fabric.backward(loss_total)
            optimizer.step()
            optimizer_auto.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_auto.zero_grad()
            torch.cuda.empty_cache()

            batch_time.update(time.time() - end)
            end = time.time()
            momentum_update(model, anchor_model, momentum=0.95)

            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            ent_losses.update(loss_ent.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(all_dataloader)}]'
                         f' | Dataset: [{cfg.dataset} - {cfg.prompt}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Entropy Loss [{ent_losses.val:.4f} ({ent_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

            loss_logger = {"Dice Loss": dice_losses.avg, "IoU Loss": iou_losses.avg, "Total Loss": total_losses.avg}
            fabric.log_dict(loss_logger)
            torch.cuda.empty_cache()
                
            num_images = transformed_images.size(0)
            with torch.no_grad():
                transformed_images = transform_auto(images_test[0], basenames[0]) 
                transformed_images = transformed_images.unsqueeze(0)  
                _, pred_masks, ious, _ = model(transformed_images, prompts_refined)
                for pred_mask, gt_mask, basename, iou in zip(pred_masks, gt_masks_test, basenames, ious):
                    pred_mask = (pred_mask > 0.5).float()
                    gt_mask = (gt_mask > 0.5).float()

            batch_dice, batch_assd, batch_hd95 = calculate_metrics(pred_mask, gt_mask)
            dice_scores.update(batch_dice, num_images)
            assd_scores.update(batch_assd, num_images)
            hd95_scores.update(batch_hd95, num_images)

            pred_mask_np = pred_mask.squeeze().cpu().numpy()  
            pred_mask_np = (pred_mask_np * 255).astype(np.uint8)  
            save_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-pred_masks", f"{basename}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray(pred_mask_np).save(save_path)

            case_results.append({
                "basename": basename, 
                "Dice": batch_dice, 
                "ASSD": batch_assd, 
                "HD95": batch_hd95
            })
            fabric.print(f"{basename} Dice: {batch_dice:.4f} - Batch ASSD: {batch_assd:.4f} - Batch HD95: {batch_hd95:.4f}")

    # --- 结尾统计代码完全保留 ---
    fabric.print(f'Test Ending...: Mean Dice: [{dice_scores.avg:.4f}] -- Mean ASSD: [{assd_scores.avg:.4f}] -- Mean HD95: [{hd95_scores.avg:.4f}]')
    case_results.append({
        "basename": "Average",  
        "Dice": dice_scores.avg,
        "ASSD": assd_scores.avg,
        "HD95": hd95_scores.avg
    })

    df = pd.DataFrame(case_results)
    if fabric.global_rank == 0:
        csv_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-test-results.csv")
        df.to_csv(csv_path, index=False)

    state = {"model": model, "optimizer": optimizer}
    fabric.save(os.path.join(cfg.out_dir, "save-ckpt", f"{cfg.dataset}-{cfg.prompt}-last-ckpt.pth"), state)


#     return optimizer, scheduler
def configure_opt(cfg: Box, model: Model):
    def lr_lambda(step):
        # Decay rate for each step (0.999 for multiplicative decay)
        decay_rate = 0.999
        
        # Apply multiplicative decay at every step
        return decay_rate ** step

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.name = corrupt
        torch.cuda.empty_cache()
        main(cfg)


def multi_main(cfg):
    prompts = ["box", "point"]
    for prompt in prompts:
        cfg.prompt = prompt
        torch.cuda.empty_cache()
        main(cfg)


def main(cfg: Box, ckpt: str = None) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name=f"{cfg.dataset}-{cfg.prompt}")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        cfg_dict = cfg.to_dict()
        os.makedirs(os.path.join(cfg.out_dir, "configs"), exist_ok=True)
        cfg_dict_path = os.path.join(cfg.out_dir, "configs", f"{cfg.dataset}-{cfg.prompt}.yaml")
        with open(cfg_dict_path, "w") as file:
            yaml.dump(cfg_dict, file)

        os.makedirs(os.path.join(cfg.out_dir, "save-ckpt"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-training.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        
        # model.setup()
        model.setup_pm()
        LoRA_Sam(model.model, 4)
        anchor_model = copy_model(model)

    load_datasets = call_load_dataset(cfg)
    all_data, train_data, val_data, test_data = load_datasets(cfg, model.model.image_encoder.img_size)
    optimizer, scheduler = configure_opt(cfg, model.model)

    fabric.print(f"All Data: {len(all_data) * cfg.batch_size}; Train Data: {len(train_data) * cfg.batch_size}; Val Data: {len(val_data) * cfg.val_batchsize}; Val Data: {len(test_data) * 1}")
    num_iters = len(train_data) * cfg.batch_size
    if ckpt is not None:
        full_checkpoint = fabric.load(ckpt)
        model.load_state_dict(full_checkpoint["model"])
        # optimizer.load_state_dict(full_checkpoint["optimizer"])
    all_data = fabric._setup_dataloader(all_data)
    # train_data = fabric._setup_dataloader(train_data)
    # val_data = fabric._setup_dataloader(val_data)
    # test_data = fabric._setup_dataloader(test_data)
    model, optimizer = fabric.setup(model, optimizer)

    # validate_med(fabric, cfg, anchor_model, all_data, name=cfg.name, iters=0)

    train_sam(cfg, fabric, model, anchor_model, optimizer, scheduler, all_data, num_iters)

    del model, anchor_model, train_data, val_data

def load_config(cfg_path: str, args: argparse.Namespace):
    config_module = importlib.import_module(cfg_path.replace('.py', '').replace('/', '.'))

    cfg = Box(config_module.config)
    if hasattr(config_module, 'base_config'):
        cfg.merge_update(config_module.base_config)
    if args.dataset:
        cfg.dataset = args.dataset
    if args.prompt:
        cfg.prompt = args.prompt
    if args.gpu_ids:
        cfg.gpu_ids = args.gpu_ids
    cfg.out_dir = "output/tokentta/"+args.dataset

    return cfg
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多卡
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁止使用cudnn的自动优化

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--cfg', type=str, required=True, help='Path to the configuration Python file (e.g., configs/config.py)')
    parser.add_argument('--dataset', type=str, help='Dataset name to override the config file')
    parser.add_argument('--prompt', type=str, help='Prompt type to override the config file')
    parser.add_argument('--gpu_ids', type=str, help='Gpu IDs to override the config file')
    args = parser.parse_args()
    cfg = load_config(args.cfg, args)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    main(cfg)
    torch.cuda.empty_cache()
    set_seed(1337)
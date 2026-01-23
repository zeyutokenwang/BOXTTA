import torch
import torch.nn as nn
import torch.nn.functional as F
from math import comb


class LearnableBezierTransform(nn.Module):
    """
    Structure-aware SBCT:
    - Intensity branch
    - Gradient (Sobel) branch
    - Structure (Laplacian) branch
    """

    def __init__(self):
        super().__init__()

        # ===== Bezier control points for 3 branches =====
        self.cp_intensity = nn.Parameter(torch.tensor([0.0, 0.33, 0.66, 1.0]))
        self.cp_gradient  = nn.Parameter(torch.tensor([0.0, 0.20, 0.80, 1.0]))
        self.cp_structure = nn.Parameter(torch.tensor([0.0, 0.25, 0.75, 1.0]))

        # ===== Fixed Sobel kernels =====
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        )

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

        # ===== Fixed Laplacian kernel =====
        laplacian = torch.tensor(
            [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0]], dtype=torch.float32
        )
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3))

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, x, base_name):
        """
        输入:
            x (torch.Tensor): [1, H, W] 或 [3, H, W]，值范围 [0,1]

        输出:
            torch.Tensor:
                - 单通道输入 -> [3, H, W]（强度 / 梯度 / 结构）
                - 三通道输入 -> [3, H, W]（保持原逻辑）
        """
        assert x.dim() == 3, "输入张量应为 3 维，形状为 [C, H, W]"
        c, h, w = x.shape
        assert c in [1, 3], "仅支持单通道 [1, H, W] 或三通道 [3, H, W] 输入"

        # Bezier 控制点（保持与你原来一致）
        cp1 = torch.sigmoid(self.cp_intensity)  # 强度
        cp2 = torch.sigmoid(self.cp_gradient)  # 梯度
        cp3 = torch.sigmoid(self.cp_structure)  # 结构

        # ======================================================
        # 情况 1：单通道输入 —— Structure-aware SBCT
        # ======================================================
        if c == 1:
            # ---------- 强度分支（原始灰度） ----------
            intensity = x  # [1,H,W]
            f_intensity = self.bezier_curve(cp1, intensity)  # [1,H,W]

            # ---------- 梯度分支（Sobel） ----------
            x_4d = x.unsqueeze(0)  # [1,1,H,W]
            gx = F.conv2d(x_4d, self.sobel_x, padding=1)
            gy = F.conv2d(x_4d, self.sobel_y, padding=1)
            grad = torch.sqrt(gx ** 2 + gy ** 2)              # [1,1,H,W]
            grad = self.normalize(grad).squeeze(0)            # [1,H,W]
            f_grad = self.bezier_curve(cp2, grad)

            # ---------- 结构分支（Laplacian） ----------
            lap = F.conv2d(x_4d, self.laplacian, padding=1)
            lap = torch.abs(lap)
            lap = self.normalize(lap).squeeze(0)              # [1,H,W]
            f_struct = self.bezier_curve(cp3, lap)

            # ---------- 拼接为 3 通道 ----------
            output = torch.cat([f_intensity, f_grad, f_struct], dim=0)  # [3,H,W]

            return output

        # ======================================================
        # 情况 2：三通道输入 —— 完全保持你原来的逻辑
        # ======================================================
        else:  # c == 3
            out_c0 = self.bezier_curve(cp1, x[0:1])  # [1, H, W]
            out_c1 = self.bezier_curve(cp2, x[1:2])  # [1, H, W]
            out_c2 = self.bezier_curve(cp3, x[2:3])  # [1, H, W]

            output = torch.cat([out_c0, out_c1, out_c2], dim=0)  # [3, H, W]
            return output

    # --------------------------------------------------
    # Bezier curve (unchanged)
    # --------------------------------------------------
    def bezier_curve(self, control_points, x):
        P0, P1, P2, P3 = control_points
        t = x
        omt = 1 - t

        B0 = omt ** 3
        B1 = 3 * t * omt ** 2
        B2 = 3 * t ** 2 * omt
        B3 = t ** 3

        f = B0 * P0 + B1 * P1 + B2 * P2 + B3 * P3
        return torch.clamp(f, 0.0, 1.0)

    @staticmethod
    def normalize(x, eps=1e-6):
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

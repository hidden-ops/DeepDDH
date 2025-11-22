import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapLoss(nn.Module):
    def __init__(self, loss_type="mse", omega=10.0, epsilon=1.0, theta=0.5, alpha=2.0):
        """
        热图损失函数，支持多种损失类型。

        参数:
            loss_type (str): 损失类型，可选 "mse", "smooth_l1", "wing", "adaptive_wing"。
            omega (float): Wing Loss 和 Adaptive Wing Loss 的参数。
            epsilon (float): Wing Loss 和 Adaptive Wing Loss 的参数。
            theta (float): Adaptive Wing Loss 的参数。
            alpha (float): Adaptive Wing Loss 的参数。
        """
        super(HeatmapLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.omega = torch.tensor(omega, dtype=torch.float32)
        self.epsilon = torch.tensor(epsilon, dtype=torch.float32)
        self.theta = torch.tensor(theta, dtype=torch.float32)
        self.alpha = torch.tensor(alpha, dtype=torch.float32)

        if self.loss_type not in ["mse", "smooth_l1", "wing", "adaptive_wing"]:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, pred_heatmap, target_heatmap):
        """
        计算热图损失。

        参数:
            pred_heatmap (Tensor): 预测热图，形状为 (batch_size, num_keypoints, H, W)。
            target_heatmap (Tensor): 目标热图，形状为 (batch_size, num_keypoints, H, W)。
        """
        if self.loss_type == "mse":
            return self.mse_loss(pred_heatmap, target_heatmap)
        elif self.loss_type == "smooth_l1":
            return self.smooth_l1_loss(pred_heatmap, target_heatmap)
        elif self.loss_type == "wing":
            return self.wing_loss(pred_heatmap, target_heatmap)
        elif self.loss_type == "adaptive_wing":
            return self.adaptive_wing_loss(pred_heatmap, target_heatmap)

    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def smooth_l1_loss(self, pred, target):
        return F.smooth_l1_loss(pred, target)

    def wing_loss(self, pred, target):
        diff = torch.abs(pred - target)
        C = self.omega - self.omega * torch.log(1 + self.omega / self.epsilon).to(pred.device)
        loss = torch.where(diff < self.omega,
                           self.omega * torch.log(1 + diff / self.epsilon),
                           diff - C)
        return loss.mean()

    def adaptive_wing_loss(self, pred, target):
        diff = torch.abs(pred - target)
        delta = diff / self.theta
        A = self.omega * (1 / (1 + delta ** self.alpha)).to(pred.device)
        C = self.omega - A * self.theta
        loss = torch.where(diff < self.theta,
                           A * diff,
                           diff - C)
        return loss.mean()

# 示例用法
if __name__ == "__main__":
    batch_size = 2
    num_keypoints = 5
    H, W = 64, 64

    # 创建随机预测热图和目标热图
    pred_heatmap = torch.randn(batch_size, num_keypoints, H, W)
    target_heatmap = torch.randn(batch_size, num_keypoints, H, W)

    # 使用 MSE 损失
    loss_fn = HeatmapLoss(loss_type="mse")
    mse_loss = loss_fn(pred_heatmap, target_heatmap)
    print(f"MSE Loss: {mse_loss.item()}")

    # 使用 Smooth L1 损失
    loss_fn = HeatmapLoss(loss_type="smooth_l1")
    smooth_l1_loss = loss_fn(pred_heatmap, target_heatmap)
    print(f"Smooth L1 Loss: {smooth_l1_loss.item()}")

    # 使用 Wing Loss
    loss_fn = HeatmapLoss(loss_type="wing")
    wing_loss = loss_fn(pred_heatmap, target_heatmap)
    print(f"Wing Loss: {wing_loss.item()}")

    # 使用 Adaptive Wing Loss
    loss_fn = HeatmapLoss(loss_type="adaptive_wing")
    adaptive_wing_loss = loss_fn(pred_heatmap, target_heatmap)
    print(f"Adaptive Wing Loss: {adaptive_wing_loss.item()}")
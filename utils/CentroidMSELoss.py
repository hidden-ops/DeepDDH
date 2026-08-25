import torch
import torch.nn as nn
import torch.nn.functional as F


class CentroidMSELoss(nn.Module):
    def __init__(self, n_class):
        """
        基于质心的 MSE 损失函数。

        参数:
            n_class (int): 类别的总数。
        """
        super(CentroidMSELoss, self).__init__()
        self.n_class = n_class

    def forward(self, pred_mask, true_mask):
        """
        计算预测图和真实图的质心，并通过 MSE 最小化质心间距离。

        参数:
            pred_mask (Tensor): 预测图，形状为 (batch_size, n_class, H, W)。
            true_mask (Tensor): 真实图，形状为 (batch_size, H, W)。
        """
        batch_size, n_class, H, W = pred_mask.shape
        device = pred_mask.device

        # 1. 对预测图应用 softmax，获取每个像素的类别概率
        pred_prob = F.softmax(pred_mask, dim=1)

        # 2. 获取每个像素的类别预测
        pred_class = torch.argmax(pred_prob, dim=1)  # 形状为 (batch_size, H, W)

        # 3. 将预测类别转换为 one-hot 编码
        pred_mask_onehot = F.one_hot(pred_class, num_classes=self.n_class).permute(0, 3, 1, 2).float()

        # 4. 将真实图转换为 one-hot 编码
        true_mask_onehot = F.one_hot(true_mask, num_classes=self.n_class).permute(0, 3, 1, 2).float()

        # 初始化质心存储
        pred_centroids = torch.zeros(batch_size, n_class, 2, device=device)
        true_centroids = torch.zeros(batch_size, n_class, 2, device=device)

        # 计算每个类别的质心
        for c in range(n_class):
            # 预测图的质心
            pred_class_map = pred_mask_onehot[:, c, :, :]  # 获取第 c 类的预测掩码
            pred_y, pred_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                                            indexing="ij")
            pred_y = pred_y.float()
            pred_x = pred_x.float()
            pred_centroid_y = (pred_class_map * pred_y).sum(dim=(1, 2)) / pred_class_map.sum(dim=(1, 2)).clamp(min=1e-6)
            pred_centroid_x = (pred_class_map * pred_x).sum(dim=(1, 2)) / pred_class_map.sum(dim=(1, 2)).clamp(min=1e-6)
            pred_centroids[:, c, 0] = pred_centroid_y
            pred_centroids[:, c, 1] = pred_centroid_x

            # 真实图的质心
            true_class_map = true_mask_onehot[:, c, :, :]  # 获取第 c 类的真实掩码
            true_y, true_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                                            indexing="ij")
            true_y = true_y.float()
            true_x = true_x.float()
            true_centroid_y = (true_class_map * true_y).sum(dim=(1, 2)) / true_class_map.sum(dim=(1, 2)).clamp(min=1e-6)
            true_centroid_x = (true_class_map * true_x).sum(dim=(1, 2)) / true_class_map.sum(dim=(1, 2)).clamp(min=1e-6)
            true_centroids[:, c, 0] = true_centroid_y
            true_centroids[:, c, 1] = true_centroid_x

        # 计算质心之间的 MSE
        mse_loss = F.mse_loss(pred_centroids, true_centroids) / 10000

        return mse_loss


# 示例用法
if __name__ == "__main__":
    batch_size, n_class, H, W = 2, 3, 10, 10
    pred_mask = torch.rand(batch_size, n_class, H, W)  # 随机生成预测图
    true_mask = torch.randint(0, n_class, (batch_size, H, W))  # 随机生成真实图

    loss_fn = CentroidMSELoss(n_class=n_class)
    loss = loss_fn(pred_mask, true_mask)
    print(f"Centroid MSE Loss: {loss.item()}")
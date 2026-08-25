import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, checkpoint_path='checkpoint.pth'):
        """
        EarlyStopping 的初始化函数。
        :param patience: 在验证性能没有提升后，允许的训练轮数。
        :param delta: 验证性能的最小改进阈值。
        :param verbose: 是否打印详细信息。
        :param checkpoint_path: 保存最佳模型的路径。
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss  # 因为越小越好，所以取负值
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        保存当前最佳模型。
        """
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss
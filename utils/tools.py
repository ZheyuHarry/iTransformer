import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# 这个后端适合在服务器端绘图
plt.switch_backend('agg')


def adjust_learning_rate(optimizer , epoch , args):
    """
        这个函数我们会根据设置和epoch轮数修改学习率
    """
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self , patience=7 , verbose=False , delta=0):
        """
            初始化参数，patience是忍耐模型不进步的轮数，delta是最少要求的进步量
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self , val_loss , model , path):
        """
            调用这个EarlyStopping类实例来判断是否这一轮需要停止训练了
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss , model , path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self , val_loss, model , path):
        """
            保存检查点
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict() , path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScalar():
    """
        提供了标准化和逆标准化的方法
    """
    def __init__(self , mean , std):
        self.mean = mean
        self.std = std

    
    def transform(self , data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self , data):
        return data * self.std + self.mean


def visual(true , preds = None , name='./pic/test.pdf'):
    """
        结果可视化

        注意这里的true参数，不是Bool值，而是表示time series的Groundtruth
    """
    plt.figure()
    plt.plot(true , label = "GroundTruth" , linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name , bbox_inches="tight")


def adjustment(gt, pred):
    """
    主要用于Anomaly Detection， 这个的目的是为了让gt和pred都为异常的时候，不要前后漏掉异常区间 ， 例如:
    gt   = [0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 0] ==>gt   = [0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 0]
    pred = [0 , 0 , 0 , 0 , 1 , 0 , 0 , 1 , 0 , 0] ==>pred = [0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 0]
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = Trues
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    # 说明这里gt是故障，但是预测出来不是，需要调整
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """
        计算预测准确率
    """
    return np.mean(y_pred == y_true)

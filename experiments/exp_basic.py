"""
    设置实验类对象，主要用于实验设定，包括模型设定，损失和优化器设定，训练测试函数编写，这个是基类函数
"""
import os
import torch
from model import iTransformer

class Exp_Basic(object):
    def __init__(self , args):
        self.args = args
        self.model_dict = {"iTransformer" : iTransformer}
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
            根据args设置修改环境变量，然后设定模型训练的设备
        """
        if self.args.use_gpu: # 如果设定说要使用gpu进行训练
            # 限制程序可以看得到的gpu
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device
    
    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        
        pass

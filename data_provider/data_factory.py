from data_provider.data_loader import Dataset_ETT_hour , Dataset_ETT_minute , Dataset_Custom , Dataset_PEMS , Dataset_Pred , Dataset_Solar

from torch.utils.data import DataLoader
# Pytorch的数据集的构造基本都是先构造数据集，然后用DataLoader来设置Batch，然后一个批量一个批量的来给你的网络输入数据

# 这个是对应地数据文件名和数据集的映射
data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "Solar": Dataset_Solar,
    "PEMS": Dataset_PEMS,
    "custom": Dataset_Custom,
}


def data_provider(args , flag):
    """
        根据给定的设定和模式，选择对应的数据集并且构造出对应的训练/测试/验证数据集，返回构造的数据集和数据加载器
    """
    Data = data_dict[args.data] # 根据设定的数据寻找对应的数据集
    timeenc = 0 if args.embed != 'timeF' else 1 # 根据设定看需要提取手动的还是特定设置的时间特征

    if flag == "test":
        shuffle_flag = False # 测试阶段需要顺序处理数据
        drop_last = True  # 如果数据集的大小不能被 batch_size 整除，丢弃最后一批数据。这么做是出于评估阶段的一致性考虑。
        batch_size = 1 # 注意评估每一个样本
        freq = args.freq # 设定好的频率字符串
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False # 预测模式下每一个样本都要预测
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred # 预测的时候就不管别的，需要专门用预测数据集模式
    else:
        shuffle_flag = True # 训练或者验证阶段是可以打乱样本顺序的
        drop_last = True
        batch_size = args.batch_size # 这个时候就根据设定来，不需要一个一个预测过去了
        freq = args.freq
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len , args.label_len , args.pred_len],
        features=args.features, # 设定需要几个特征
        target = args.target,
        timeenc=timeenc,
        freq=freq
    )

    print(f"*****In data_factory.data_provider , flag = {flag} , len(data_set) = {len(data_set)}*****")
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set , data_loader
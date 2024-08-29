"""
    设置各种metric指标，简介如下：
    1. RSE (Relative Squared Error)
    定义: 相对平方误差，用于衡量预测值与真实值之间的误差相对于真实值的变异性。
    计算方式: 计算预测与真实值之间的平方差的平方根，然后与真实值的方差的平方根进行比较。
    用途: 评估模型的预测能力，尤其在数据的范围很大时。

    2. CORR (Correlation Coefficient)
    定义: 相关系数，衡量预测值与真实值之间的线性关系强度。
    计算方式: 通过计算协方差与标准差的乘积来获得。
    用途: 判断预测结果与真实数据之间的相关性，值范围从 -1 到 1。

    3. MAE (Mean Absolute Error)
    定义: 平均绝对误差，表示预测值与真实值之间差异的平均值。
    计算方式: 计算预测值与真实值之差的绝对值的平均。
    用途: 直观地衡量误差大小，适用于对异常值不敏感的情况。

    4. MSE (Mean Squared Error)
    定义: 平均平方误差，测量预测值与真实值之间差异的平方的平均。
    计算方式: 计算预测值与真实值之差的平方的平均。
    用途: 敏感于异常值，常用于回归分析中。

    5. RMSE (Root Mean Squared Error)
    定义: 均方根误差，是均方误差的平方根。
    计算方式: 计算 MSE 的平方根。
    用途: 以原始单位表示误差，更直观地反映预测的精度。

    6. MAPE (Mean Absolute Percentage Error)
    定义: 平均绝对百分比误差，表示预测误差相对于真实值的百分比。
    计算方式: 计算预测值与真实值之差的绝对值与真实值的比值的平均。
    用途: 衡量预测的相对误差，适用于希望了解误差相对大小的场合。
    
    7. MSPE (Mean Squared Percentage Error)
    定义: 平均平方百分比误差，预测误差相对于真实值的平方的平均。
    计算方式: 计算预测值与真实值之差的平方与真实值的比值的平均。
    用途: 能够强调较大的误差，适用于对高误差敏感的情况。
"""

import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

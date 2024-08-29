"""
    这个文件是定义了一些时间特征类，提供了不同粒度的时间特征，可以被用于机器学习模型，以帮助模型理解时间的周期性和规律性。
"""

from typing import List

import numpy as np
import pandas as pd

# offsets是时间偏移量，to_offset是把频率字符串如"1H"转化为真实的时间偏移量
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

# 这个是时间特征的基类，
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self , index:pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


"""
#####################################################

    下面这些都是常用的时间特征，表示大概过了多少分钟，过了一年之内的几个月了，这些都是时间特征，每个特征都反映了时间的不同维度，有助于捕捉时间序列数据中的周期性和趋势。
    
    然后由于秒不可能有60(自动进位)，所以不用-1，像月份就需要

    然后由于原来是[0 , 1] ， 现在给向下裁剪到[-0.5, 0.5]

#####################################################    
"""


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
        根据提供的频率字符串返回显式的TimeFeature列表

        这个频率参数是[数字][粒度]的形式，例如: 12H , 3D , 5min , etc.
    """

    # 这里的key代表了各种各样的时间偏移类别，后面的value代表了与之相关的时间特征类
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    # 将偏移字符串转化为时间偏移量
    offset = to_offset(freq_str)

    # 遍历上面的字典，找到与当前偏移量类型相同的偏移类别，然后返回这些相关的偏移类别的实例
    for offset_type , feature_classes in features_by_offsets.items():
        if isinstance(offset , offset_type):
            return [cls() for cls in feature_classes]

    # 到这里没有return说明这个偏移字符串虽然是正确的格式，但是不被这里的算法支持，要raise一个RE

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates , freq="h"):
    """
        这个函数用来计算这个日期的相关时间特征，计算哪方面的特征由我们手动指定偏移字符串"freq"
    """
    return np.vstack([feature(dates) for feature in time_features_from_frequency_str(freq)])
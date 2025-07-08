import numpy as np

def drop_outlier(array, count, bins):
    """
    使用2-σ准则从数据序列中移除异常值。
    此函数改编自提供的notebook。

    参数:
        array (np.array): 数据数组。
        count (int): 数据点数量。
        bins (int): 将数据分成的区间数。

    返回:
        np.array: 非异常值数据点的索引数组。
    """
    index = []
    range_ = np.arange(1, count, bins)
    for i in range_[:-1]:
        array_lim = array[i:i + bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max, th_min = mean + sigma * 2, mean - sigma * 2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)

import os
import time
import spectral.io.envi as e
import numpy as np
from pprint import pprint
from scipy.signal import savgol_filter


def label_mapping(label_arr):
    """
    :param label_arr: 2D label array
    :return: 2D label array mapped to [1, num_classes]
    """
    # 获取所有不重复的标签值
    unique_labels = np.unique(label_arr)
    print('unique labels:', unique_labels)

    # 创建一个映射字典，将原始标签映射为连续的整数
    label_mapping_dict = {label: idx + 1 for idx, label in enumerate(unique_labels)}

    # 对标签数组进行映射
    return np.vectorize(label_mapping_dict.get)(label_arr)


def remove_envelope(spectrum, window_size=11, polyorder=3):
    """
    从光谱中去除包络线
    Args:
        spectrum (ndarray): 输入的光谱，形状为 (C,)
        window_size (int): Savitzky-Golay平滑窗口大小
        polyorder (int): Savitzky-Golay多项式拟合阶数
    Returns:
        ndarray: 去除包络线后的光谱
    """
    smoothed = savgol_filter(spectrum, window_size, polyorder)
    envelope = np.maximum(smoothed, spectrum)
    return np.float32(spectrum) / np.float32(envelope)


def remove_envelope_3d(array, window_size=11, polyorder=3):
    """
    从3D数组中的每个光谱去除包络线
    Args:
        array (ndarray): 输入的3D数组，形状为 (H, W, C)
        window_size (int): Savitzky-Golay平滑窗口大小
        polyorder (int): Savitzky-Golay多项式拟合阶数
    Returns:
        ndarray: 去除包络线后的3D数组
    """
    h, w, c = array.shape
    processed_array = np.zeros_like(array)
    for i in range(h):
        for j in range(w):
            processed_array[i, j, :] = remove_envelope(array[i, j, :], window_size, polyorder)
    return processed_array


if __name__ == '__main__':

    fn = r"H:\thesis\Jabal_Damkh\Jabal_Damkh_partial_labelraster_clip.hdr"
    o_fn = r"H:\thesis\Jabal_Damkh\Jabal_Damkh_partial_labelraster_clip_mod.hdr"
    label_fn = r"H:\thesis\Jabal_Damkh\Jabal_Damkh_partial_label_clip_mod.hdr"

    raster = e.open(fn)
    new_meta = raster.metadata.copy()
    nl, ns, nb_with_label = raster.shape
    nb = nb_with_label - 1

    new_data = np.zeros((nl, ns, nb_with_label), dtype=np.float32)

    data = raster.read_bands(range(nb))
    first_band = raster.read_band(0)
    background = (first_band == 0)
    label = raster.read_band(nb_with_label-1)

    enhanced_data = remove_envelope_3d(data)
    mapped_label = label_mapping(label)

    mapped_label[background] = 0
    new_data[:, :, 0:nb] = enhanced_data
    new_data[:, :, nb] = mapped_label

    e.save_image(o_fn, new_data, metadata=new_meta, interleave='bip')

    spec_keys = ['wavelength', 'fwhm', 'wavelength units', 'data gain values', 'data offset values', 'bbl']
    for spec_key in spec_keys:
        if spec_key in new_meta:
            new_meta.pop(spec_key)
    for key in new_meta.keys():
        value = new_meta[key]
        if isinstance(value, list) and len(value) == nb:
            new_meta[key] = value[nb - 1]
        else:
            new_meta[key] = value
    e.save_classification(label_fn, mapped_label, metadata=new_meta)

import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import spectral.io.envi as e
import shutil
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    """
    Need to firstly stack the image and label raster as a single raster file (you can use ENVI band/layer stack tool)
    so that image and label can be of the rigorously same nRows and nColumns.
    Please let label as the last band.
    classes value range: [1, num_classes]
    """

    def __init__(self, raster_dir, image_transform=None, label_transform=None):
        self.raster_dir = raster_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.raster_files = [f for f in os.listdir(raster_dir) if f.endswith('.hdr')]

    def __len__(self):
        return len(self.raster_files)

    def __getitem__(self, idx):
        raster_name = os.path.join(self.raster_dir, self.raster_files[idx])
        raster = e.open(raster_name)
        meta = raster.metadata
        nl, ns, nb = raster.shape

        scale = meta['reflectance scale factor'] if 'reflectance scale factor' in meta else 1.0

        image = np.array(scale * raster.read_bands(range(nb - 1)))
        label = np.array(scale * raster.read_band(nb - 1))

        """nn.CrossEntropyLoss expects the target to contain class indices in the range [0, nb_classes-1]"""
        # if np.nanmin(label) > 0:
        #     label = label - 1

        if self.image_transform:
            """
            (H, W, C) -> (C, H, W) when loaded as tensor
            """
            image = self.image_transform(image).float()

        if self.label_transform:
            label = self.label_transform(label)
            """
            shape of single label will be (1, H, W) when loaded as tensor,
            but only batches of spatial targets are supported (3D tensors) in loss function,
            the first dimension is batch size.
            thus a single label need to be 2D (H, W)
            e.g. (1, 16, 16) -> (16, 16)
            """
            label = label.squeeze(0).long()

        return image, label


def calc_dataset_stats(dataset):
    # 定义一个空列表来存储每个样本的像素值
    pixel_values = []

    # 遍历数据集，收集像素值
    for data in dataset:
        image, _ = data  # 假设数据集返回图像和标签，这里我们只关心图像
        pixel_values.append(image)

    # 将列表转换为 numpy 数组
    pixel_values = np.array(pixel_values)  # shape: (N, H, W, C)

    # 计算均值和标准差
    mean = np.mean(pixel_values, axis=(0, 1, 2))  # 计算每个通道的均值
    std = np.std(pixel_values, axis=(0, 1, 2))  # 计算每个通道的标准差

    return mean, std


def extract_patches(input_raster_uri, output_patches_dir, patch_size=(16, 16), overlap=(2, 2),
                    dtype=None):
    """
    Segment a single remote sensing image (original or augmented) or it's classification map into
    series of patches as an input dataset for deep network

    :param input_raster_uri:
    :param output_patches_dir:
    :param patch_size:
    :param overlap:
    :param dtype:
    :return:
    """

    raster = e.open(input_raster_uri)
    nl, ns, nb = raster.shape
    meta = raster.metadata
    dtype = dtype if dtype else raster.dtype
    h_patch, w_patch = patch_size
    h_overlap, w_overlap = overlap

    data = raster.load()
    patch_meta = dict(
        lines=h_patch,
        samples=w_patch,
        bands=nb
    )

    # total_patches_num = int(np.floor(nl / h_patch) * np.floor(ns / w_patch))
    current_patches_num, y_idx, x_idx = 0, 0, 0

    for y in range(0, nl - h_patch + 1, h_patch - h_overlap):
        x_idx = 0
        for x in range(0, ns - w_patch + 1, w_patch - w_overlap):
            o_fn = os.path.join(output_patches_dir, f'_{y_idx:04d}_{x_idx:04d}.hdr')
            if not os.path.exists(os.path.dirname(o_fn)):
                os.makedirs(os.path.dirname(o_fn))
            patch = data[y:y + h_patch, x:x + w_patch, :]
            e.save_image(o_fn, patch, metadata=patch_meta, dtype=dtype, interleave='bip')
            """
            PyTorch Dataloader will load images by CHW, thus the first dimension need to be bands
            """

            print(f'\rExtracted {current_patches_num + 1} patches', end='')

            current_patches_num += 1
            x_idx += 1

        y_idx += 1


def extract_patches_with_padding(input_raster_uri, output_patches_dir, patch_size=(16, 16), dtype=None):
    """
    Segment a single remote sensing image (original or augmented) or its classification map into
    series of patches as an input dataset for deep network

    :param input_raster_uri:
    :param output_patches_dir:
    :param patch_size:
    :param dtype:
    :return: No return
    """

    raster = e.open(input_raster_uri)
    nl, ns, nb = raster.shape
    dtype = dtype if dtype else raster.dtype
    h_patch, w_patch = patch_size

    data = raster.load()
    patch_meta = dict(
        lines=h_patch,
        samples=w_patch,
        bands=nb
    )

    # Calculate the number of patches in each dimension
    nl_patches = int(np.ceil(nl / h_patch))
    ns_patches = int(np.ceil(ns / w_patch))

    # Pad the image if needed to ensure all patches are covered
    pad_nl = nl_patches * h_patch - nl
    pad_ns = ns_patches * w_patch - ns
    padded_data = np.pad(data, ((0, pad_nl), (0, pad_ns), (0, 0)), mode='constant')

    # Total number of patches
    total_patches_num = nl_patches * ns_patches
    current_patches_num, y_idx, x_idx = 0, 0, 0

    for y in range(0, nl_patches * h_patch, h_patch):
        x_idx = 0
        for x in range(0, ns_patches * w_patch, w_patch):
            o_fn = os.path.join(output_patches_dir, f'_{y_idx:04d}_{x_idx:04d}.hdr')
            if not os.path.exists(os.path.dirname(o_fn)):
                os.makedirs(os.path.dirname(o_fn))
            patch = padded_data[y:y + h_patch, x:x + w_patch, :]
            e.save_image(o_fn, patch, metadata=patch_meta, dtype=dtype)
            print(f'\rExtracted {current_patches_num + 1} patches', end='')
            current_patches_num += 1
            x_idx += 1

        y_idx += 1

    print("\nExtraction finished.")


def reconstruct_image(patches, nl, ns, patch_size=(16, 16)):
    """
    Reconstruct the original image from a series of patches without overlap

    :param patches: Predicted array of shape (n_patches, h_patch, w_patch)
    :param nl: Number of lines (rows) in the original image
    :param ns: Number of samples (columns) in the original image
    :param patch_size: Size of each patch (height, width)
    :return: Reconstructed image
    """
    h_patch, w_patch = patch_size

    # Calculate the number of patches in each dimension
    nl_patches = int(np.ceil(nl / h_patch))
    ns_patches = int(np.ceil(ns / w_patch))

    # Initialize an array to hold the reconstructed image
    reconstructed = np.zeros((nl_patches * h_patch, ns_patches * w_patch), dtype=patches.dtype)

    idx = 0
    for y in range(0, nl_patches * h_patch, h_patch):
        for x in range(0, ns_patches * w_patch, w_patch):
            # Add the patch to the reconstructed image
            reconstructed[y:y + h_patch, x:x + w_patch] = patches[idx, :, :]
            idx += 1

    # Trim the padded regions to get the final reconstructed image
    reconstructed = reconstructed[:nl, :ns]

    return reconstructed


def divide_dataset(source_dir, target_dir):
    """
    divide dataset(extracted patches) into 2 parts: train and valid
    :param source_dir:
    :param target_dir:
    :return:
    """
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'valid')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # 获取目标文件列表
    fn_list = [f for f in os.listdir(source_dir) if f.endswith('.hdr')]

    # 设置抽选比例和数量
    total_num = len(fn_list)
    test_proportion = 0.2  # 要移动的文件比例
    test_num = int(total_num * test_proportion)

    # 随机抽选要移动的文件
    test_list = random.sample(fn_list, test_num)
    train_list = [f for f in fn_list if f not in test_list]

    # 移动文件到目标目录
    for _ in train_list:
        source_path = os.path.join(source_dir, _)
        train_path = os.path.join(train_dir, _)
        shutil.copy(source_path, train_path)
        shutil.copy(source_path.replace('.hdr', '.img'), train_path.replace('.hdr', '.img'))

    for _ in test_list:
        source_path = os.path.join(source_dir, _)
        test_path = os.path.join(test_dir, _)
        shutil.copy(source_path, test_path)
        shutil.copy(source_path.replace('.hdr', '.img'), test_path.replace('.hdr', '.img'))


if __name__ == '__main__':
    # image_fn = r"H:\thesis\Cuprite\Cuprite_S1_R188\Cuprite_S1_R188.hdr"
    # label_fn = r"H:\thesis\Cuprite\Cuprite_S1_R188\Cuprite_S1_R188_SAM_classified.hdr"
    raster_fn = r"H:\thesis\Jabal_Damkh\Jabal_Damkh_partial_labelraster_clip_mod.hdr"
    output_dir = r"H:\thesis\Jabal_Damkh\dataset"
    extract_patches_with_padding(raster_fn, os.path.join(output_dir, 'total'), patch_size=(16, 16))
    divide_dataset(os.path.join(output_dir, 'total'), output_dir)

# class MyDataset(Dataset):
#     def __init__(self, image_dir, label_dir, transform=None):
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.hdr')])
#         self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.hdr')])
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.image_dir, self.image_files[idx])
#         label_name = os.path.join(self.label_dir, self.label_files[idx])
#
#         image = np.array(e.open(img_name).load())
#         label = np.array(e.open(label_name).load())
#
#         """nn.CrossEntropyLoss expects the target to contain class indices in the range [0, nb_classes-1]"""
#         if np.nanmin(label) > 0:
#             label = label - 1
#
#         if self.transform:
#             image = self.transform(image)
#             label = self.transform(label)
#
#         return image, label

# def reconstruct_image(input_patches_dir, output_raster_uri, patch_size=(64, 64), overlap=(2, 2)):
#     """
#     Reconstruct an image from a series of patches
#
#     :param input_patches_dir: Directory containing the image patches
#     :param output_raster_uri: URI for the output reconstructed raster
#     :param patch_size: Size of the patches
#     :param overlap: Amount of overlap between patches
#     """
#
#     patch_files = os.listdir(input_patches_dir)
#     if not patch_files:
#         raise ValueError("No patch files found in the input directory")
#
#     patch_files.sort()  # Ensure patches are sorted correctly
#
#     # Get patch size and number of patches
#     h_patch, w_patch = patch_size
#     h_overlap, w_overlap = overlap
#
#     num_patches_y = len([fn for fn in patch_files if fn.endswith('.hdr')])  # Assuming all patches are .hdr files
#     num_patches_x = len(patch_files) // num_patches_y
#
#     # Initialize reconstructed image
#     reconstructed_image = np.zeros((num_patches_y * (h_patch - h_overlap) + h_overlap,
#                                     num_patches_x * (w_patch - w_overlap) + w_overlap,
#                                     3), dtype=np.uint8)  # Assuming 3 bands for RGB image
#
#     # Reconstruct the image
#     for y_idx in range(num_patches_y):
#         for x_idx in range(num_patches_x):
#             patch_fn = os.path.join(input_patches_dir, f'_{y_idx:04d}_{x_idx:04d}.hdr')
#             patch = rio.open(patch_fn).read()
#             y_start = y_idx * (h_patch - h_overlap)
#             y_end = y_start + h_patch
#             x_start = x_idx * (w_patch - w_overlap)
#             x_end = x_start + w_patch
#             reconstructed_image[y_start:y_end, x_start:x_end, :] = patch
#
#     # Write reconstructed image to disk
#     with rio.open(output_raster_uri, 'w', driver='GTiff', height=reconstructed_image.shape[0],
#                   width=reconstructed_image.shape[1], count=reconstructed_image.shape[2],
#                   dtype=reconstructed_image.dtype) as dst:
#         dst.write(reconstructed_image)

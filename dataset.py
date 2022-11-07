# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Realize the function of dataset preparation."""
import os
import queue
import threading

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torchvision.transforms as T

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data
__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class TrainValidDataset(Dataset):
    def __init__(self, paths_file_path, transform=T.Compose([T.ToTensor()])):
        self.path_files_names = pd.read_csv(paths_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.path_files_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lr_name = self.path_files_names.iloc[idx, 1]
        lr_image = read_xray(lr_name)
        gt_name = self.path_files_names.iloc[idx, 2]
        gt_image = read_xray(gt_name)
        lr_tensor = self.transform(lr_image)
        gt_tensor = self.transform(gt_image)
        return {"gt": gt_tensor, "lr": lr_tensor}

class TestImageDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_gt_images_dir (str): ground truth image in test image
        test_lr_images_dir (str): low-resolution image in test image
    """

    def __init__(self, paths_file_path, transform=T.Compose([T.ToTensor()])):
        self.path_files_names = pd.read_csv(paths_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.path_files_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lr_name = self.path_files_names.iloc[idx, 1]
        lr_image = read_xray(lr_name)
        gt_name = self.path_files_names.iloc[idx, 2]
        gt_image = read_xray(gt_name)
        lr_tensor = self.transform(lr_image)
        gt_tensor = self.transform(gt_image)
        return {"gt": gt_tensor, "lr": lr_tensor}


# class PrefetchGenerator(threading.Thread):
#     """A fast data prefetch generator.

#     Args:
#         generator: Data generator.
#         num_data_prefetch_queue (int): How many early data load queues.
#     """

#     def __init__(self, generator, num_data_prefetch_queue: int) -> None:
#         threading.Thread.__init__(self)
#         self.queue = queue.Queue(num_data_prefetch_queue)
#         self.generator = generator
#         self.daemon = True
#         self.start()

#     def run(self) -> None:
#         for item in self.generator:
#             self.queue.put(item)
#         self.queue.put(None)

#     def __next__(self):
#         next_item = self.queue.get()
#         if next_item is None:
#             raise StopIteration
#         return next_item

#     def __iter__(self):
#         return self


# class PrefetchDataLoader(DataLoader):
#     """A fast data prefetch dataloader.

#     Args:
#         num_data_prefetch_queue (int): How many early data load queues.
#         kwargs (dict): Other extended parameters.
#     """

#     def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
#         self.num_data_prefetch_queue = num_data_prefetch_queue
#         super(PrefetchDataLoader, self).__init__(**kwargs)

#     def __iter__(self):
#         return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import os
import csv
import torchvision.transforms as transforms

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from Dataset.spatial_transfoms import*
from Dataset.temporal_transforms import*
import random
# from spatial_transfoms import*
# from temporal_transforms import*

import logging
import sys

#from vidaug import augmentors as va


class TemporalSection(object):
    """Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        length = frame_indices.shape[0]
        downsample = max(min(int(length/self.size), self.downsample),1)
        sections = int(length/self.downsample)
        # print(sections)
        downFrames = [frame_indices[i] for i in range(0, sections*self.downsample, downsample)]
        # if length < self.size():
        length = len(downFrames)
        sections = int(length/self.size)
        # print(len(downFrames), sections)
        out = []
        # print(len(out))
        for i in range(0,sections):
            print(i*self.size, (i+1)*self.size)
            out.extend(downFrames[i*self.size:(i+1)*self.size])
        # print(len(out))
        downFrames = downFrames[sections*self.size:]
        while len(downFrames) < self.size and len(downFrames)>0:
            for index in downFrames:
                if len(downFrames) >= self.size:
                    break
                # out = np.vstack((out,index))
                downFrames.append(index)
        out.extend(downFrames)

        return out

class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        # vid_duration  = frame_indices.shape[0]
        downsample = max(min(int(self.vid_duration/self.size), self.downsample),1)
        # clip_duration = self.size * downsample

        # rand_end = max(0, vid_duration - clip_duration-1)
        # begin_index = random.randint(0, rand_end)
        end_index = min(self.begin_index + self.clip_duration, self.vid_duration)
        # print(begin_index, end_index)
        out = frame_indices[self.begin_index:end_index]
        # random_out = [random.randint(0,downsample-1) for i in range(0,self.size)]
        # print(random_out)
        # print(out.shape)
        while len(out) < self.clip_duration:
            for index in out:
                if len(out) >= self.clip_duration:
                    break
                # out = np.vstack((out,index))
                out.append(index)
            # print("new_shape: ",out.shape)
        # selected_frames = [out[i] for i in range(0, self.clip_duration, downsample)]
        selected_frames = [i for i in range(self.begin_index, self.clip_duration+self.begin_index, downsample)]
        return selected_frames
    
    def randomize_parameters(self, frame_indices):
        self.vid_duration  = len(frame_indices)
        downsample = max(min(int(self.vid_duration/self.size), self.downsample),1)
        self.clip_duration = self.size * downsample

        rand_end = max(0, self.vid_duration - self.clip_duration-1)
        self.begin_index = random.randint(0, rand_end)

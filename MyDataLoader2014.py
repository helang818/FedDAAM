import torch.utils.data as udata
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
from pydub import AudioSegment
import random
import logging

normalVideoShape = 610
normalAudioShape = 186

class temporalMask():
    def __init__(self, drop_ratio):
        self.ratio = drop_ratio
    def __call__(self, frame_indices):
        frame_len = frame_indices.shape[0]
        sample_len = int(self.ratio*frame_len)
        sample_list = random.sample([i for i in range(0, frame_len)], sample_len)
        frame_indices[sample_list,:]=0
        return frame_indices
    
class AffectnetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset):
        # print('initial balance sampler ...')

        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        expression_count = [0] * 63 # 共8题，每题0-3分
        for idx in self.indices:
            label = dataset.label[idx]
            expression_count[int(label)] += 1

        self.weights = torch.zeros(self.num_samples)
        for idx in self.indices:
            label = dataset.label[idx]
            self.weights[idx] = 1. / expression_count[int(label)]

        # print('initial balance sampler OK...')


    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))


    def __len__(self):
        return self.num_samples

class MyDataLoader(udata.Dataset):
    def __init__(self, videoFileName, AudioFileName, videoFileNameR, AudioFileNameR, type) -> None:
        super().__init__()
        labelPath = "/home/ywz/AVEC2014_data/DepressionLabels/Labels/DepressionLabels"
        if type == "train":
            self.temp = temporalMask(0)
        else :
            self.temp = None
        
        self.videoList = []
        self.audioList = []
        self.label = []
        self.videoListR = []
        self.audioListR = []
        self.labelR = []
        self.gender = []
        self.diagnose = []

        genderDf = pd.read_csv("/home/ywz/code/2014Gender.csv") 
        dff = genderDf[['file', 'gender']]
        dff.set_index(keys='file', inplace=True)
        dff = dff.T
        genderDict = dff.to_dict(orient='records')[0]

        for file in os.listdir(videoFileName):
            self.videoList.append(os.path.join(videoFileName, file))
            self.audioList.append(os.path.join(AudioFileName, file.replace("video", "audio")))
            file = file.replace("Freeform", "Northwind")
            self.videoList.append(os.path.join(videoFileName, file))
            self.audioList.append(os.path.join(AudioFileName, file.replace("video", "audio")))

            # assert np.isnan(videoData).sum() == 0, logging.info("VideoNan:{}".format())

            file_csv = pd.read_csv(os.path.join(labelPath, file.replace("_Freeform_video.npy", "_Depression.csv")))

            bdi = int(file_csv.columns[0])
            # phq8 = bdi
            
            # if bdi >= 5 and bdi<= 13:
            #     phq8 = (bdi-5)/8*4 + 5
            #     self.label.append(phq8)
            # elif bdi >= 14 and bdi <= 20:
            #     phq8 = (bdi-14)/6*4+10
            #     self.label.append(phq8)
            # elif bdi >= 21:
            #     phq8 = (bdi-21)/42*9+15
            #     self.label.append(phq8)
            # self.label.append(phq8)
            
            self.gender.append(genderDict[file.replace("npy","mp4")])
            self._label.append(bdi)
            self.diagnose.append(0)
            
    def __getitem__(self, index: int):
        videoData, audioData, label, _label = np.load(self.videoList[index]), np.load(self.audioList[index]), self.label[index], self._label[index]

        gender= self.gender[index]
        diagnose = self.diagnose[index]

        label, gender = np.array(label), np.array(gender)
        diagnose = np.array(diagnose)
        _label = np.array(_label)
        # videoData = torch.from_numpy(videoData)
        # audioData = torch.from_numpy(audioData)
        if self.temp is not None:
            videoData = self.temp(videoData)
        label = torch.from_numpy(label).type(torch.float)
        _label = torch.from_numpy(_label).type(torch.float)
        gender = torch.from_numpy(gender).type(torch.int)
        diagnose = torch.from_numpy(diagnose).type(torch.int)

        if audioData.shape[0] > normalAudioShape:
            audioData = audioData[:180,:]
         # 表达式为false时，触发assert
        assert videoData.shape[0] == normalVideoShape
        assert audioData.shape[0] <= normalAudioShape
        assert videoData.shape[0] > 0
        assert audioData.shape[0] > 0

        clip_duration = normalAudioShape
        while audioData.shape[0] < clip_duration:
            for index in audioData:
                if audioData.shape[0] >= clip_duration:
                    break
        # out = np.vstack((out,index))
                audioData = np.append(audioData, np.expand_dims(index, 0), axis=0)


        assert videoData.shape[1] == 680, "invalid file:{}-{}-{}".format(self.videoList[index])

        
        videoData = torch.from_numpy(videoData)
        
        videoData = videoData.type(torch.float)
        audioData = np.array(audioData)
        audioData = torch.from_numpy(audioData)
        audioData = audioData.type(torch.float)

        return videoData, audioData, label, gender, diagnose,  _label
    
    def __len__(self) -> int:
        return len(self.videoList)


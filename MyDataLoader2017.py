import torch.utils.data as udata
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
from pydub import AudioSegment
import random
import logging

# 610*680

normalVideoShape = 900
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
        print('initial balance sampler ...')

        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        expression_count = [0] * 24 # 共8题，每题0-3分
        for idx in self.indices:
            label = dataset.label[idx]
            expression_count[int(label)] += 1

        self.weights = torch.zeros(self.num_samples)
        for idx in self.indices:
            label = dataset.label[idx]
            self.weights[idx] = 1. / expression_count[int(label)]

        print('initial balance sampler OK...')


    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))


    def __len__(self):
        return self.num_samples

class MyDataLoader(udata.Dataset):
    def __init__(self, videoFileName, AudioFileName, type) -> None:
        super().__init__()

        if type == "train":
            labelPath = "/data1/zhaojunnan/new_FedML/2017/Lable/train.xlsx"
            #audioDurationPath = "/data2/zjn/unzip/train"
            self.temp = temporalMask(0.45)
            diagnoseDict = np.load("/data1/zhaojunnan/new_FedML/2017/Lable/train_diag.npy", allow_pickle=True).item()
        else:
            labelPath = "/data1/zhaojunnan/new_FedML/2017/Lable/dev.xlsx"
            #audioDurationPath = "/extend_disk/ywz/AVEC2016-2017/unzip/dev"
            self.temp = None
            diagnoseDict = np.load("/data1/zhaojunnan/new_FedML/2017/Lable/dev_diag.npy", allow_pickle=True).item()

        df = pd.read_excel(labelPath)
        dff = df[['Participant_ID','PHQ8_Score']]
        dff.set_index(keys='Participant_ID', inplace=True)
        dff = dff.T
        labelDict = dff.to_dict(orient='records')[0] # 标签字典 id:score
        self.label = []
        self.type = type

        dffGender = df[['Participant_ID','Gender']]
        dffGender.set_index(keys='Participant_ID', inplace=True)
        dffGender = dffGender.T
        genderDict = dffGender.to_dict(orient='records')[0]
        self.gender = []
        self.diagnose = []
        flag = 0
        self.videoList = []
        self.audioList = []
        self.audioDuration = []

        dirs = os.listdir(videoFileName)
        for dir in dirs:
            id = dir.split(".")[0].split("_")[-1]
            self.videoList.append(os.path.join(videoFileName, dir))
            self.audioList.append(os.path.join(AudioFileName, id, dir))
            
            self.label.append(labelDict[int(id)])
            self.gender.append(genderDict[int(id)])
            self.diagnose.append(diagnoseDict[id])



    def __getitem__(self, index: int):

        # videoData, audioData, label = self.videoFeature[index], self.audioFeature[index], self.label[index]
        videoData, audioData, label = np.load(self.videoList[index]), np.load(self.audioList[index]), self.label[index]

        gender = self.gender[index]
        diagnose = self.diagnose[index]
        # duration = self.audioDuration[index]

        label = np.array(label)
        gender = np.array(gender)
        diagnose = np.array(diagnose)
        # duration = np.array(duration)

        videoData = torch.from_numpy(videoData)
        audioData = torch.from_numpy(audioData)
        if self.temp is not None:
            videoData = self.temp(videoData)
        label = torch.from_numpy(label).type(torch.int)
        gender = torch.from_numpy(gender).type(torch.int)
        diagnose = torch.from_numpy(diagnose).type(torch.int)

        # 表达式为false时，触发assert
        assert videoData.shape[0] <= normalVideoShape, "invalid file:{}-{}-{}".format(self.videoList[index])
        assert audioData.shape[0] <= normalAudioShape
        assert videoData.shape[0] > 0
        assert audioData.shape[0] > 0

        if videoData.shape[0] < normalVideoShape:
            zeroPadVideo = nn.ZeroPad2d(padding=(0,0,0,normalVideoShape-videoData.shape[0]))
            videoData = zeroPadVideo(videoData)
        if audioData.shape[0] < normalAudioShape:
            zeroPadAudio = nn.ZeroPad2d(padding=(0,0,0,normalAudioShape-audioData.shape[0]))
            audioData = zeroPadAudio(audioData)
        
        videoData = videoData.type(torch.float)
        audioData = audioData.type(torch.float)
        if self.type == "train":
            return videoData, audioData, label, gender, diagnose
        if self.type == "dev":
            return videoData, audioData, label, gender, diagnose, self.videoList[index]

    def __len__(self) -> int:
        return len(self.videoList)

if __name__ == '__main__' :
    nohog = np.load("/extend_disk/ywz/AVEC2016-2017/noHOG/train/459/0_459.npy")
    print(nohog.shape)  #610 * 324







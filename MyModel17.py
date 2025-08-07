
from collections import OrderedDict
import torch.nn as nn
import vilbertModel.vilbert.vilbert as vilbert
import torch
import numpy as np
import logging
from torch.autograd import Function
from TCNModel import TCNModel 
#from MDNModel import MDN

#自定义求导规则
#GRL层代码 将传到本层的误差乘以一个负数(-alpha)，这样就会使得GRL前后的网络其训练目标相反，以实现对抗的效果。
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

#grad_output.neg()是在进行梯度翻转 将原梯度变为负数
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None




class ConvNet1d(nn.Module):
    def __init__(self) -> None:
        super(ConvNet1d, self).__init__()
        self.fc = nn.Linear(256, 128)
    
    def forward(self,input):
        sizeTmp = input.size(1)
        batch_size = input.size(0)
        outConv1d = input.contiguous().view(input.size(0)*input.size(1),-1)
        output = self.fc(outConv1d)
        output = output.view(batch_size, sizeTmp, -1)

        return output

class TransformNet(nn.Module):
    def __init__(self, config):
        super(TransformNet, self).__init__()
        self.model1 = vilbert.BertModel(config)
    
    def forward(self, inputVideo, inputAudio):
        output = self.model1(inputVideo, inputAudio)
        return output


# 域对抗分类器
class AdModel(nn.Module):
    def __init__(self) -> None:
        super(AdModel, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
    def forward(self, input, alpha):
        reverse_feature = ReverseLayerF.apply(input, alpha)
        domain_output = self.domain_classifier(reverse_feature)
        return domain_output

class Regress(nn.Module):
    def __init__(self) -> None:
        super(Regress, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU())

        self.f1 =  nn.Sequential(
            nn.Linear(130, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        
    
    def forward(self, x, gender, diagnose):
        x = x.view(-1, 2048) #激活函数
        x = self.fc(x)
        x = torch.cat((x, gender.float()),dim=1)
        x = torch.cat((x, diagnose.float()),dim=1)
        return self.f1(x)

class AttentionModel(nn.Module):
    def __init__(self) -> None:
        super(AttentionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, x):
        x = x.view(-1,2048)
        x = self.fc(x)
        return x

class Net(nn.Module):
    def __init__(self, bertConfig,device) -> None:
        super().__init__()
        self.TCNModel = TCNModel() 
        self.Conv1dModel = ConvNet1d()
        self.TransformerModel = TransformNet(bertConfig)
        self.AdModel = AdModel()
        self.Regress = Regress()
        self.AttentionModel = AttentionModel()
        

    def forward(self,inputVideo,inputAudio, gender, diagnose, alpha): 
        # TCN 
        inputVideo = self.TCNModel(inputVideo)
        # 调整维度维128
        outputConv1dVideo = self.Conv1dModel(inputVideo)
        # vilBert 音视频交互 
        output = self.TransformerModel(outputConv1dVideo, inputAudio)
        output1, output2 = output[2], output[3]
        outputFeature = torch.cat((output1, output2),dim=1) 

        # 性别、是否被诊断为抑郁
        _gender = gender
        _diagnose = diagnose
        gender = gender * outputFeature.shape[0]
        gender = gender[:, np.newaxis]
        diagnose = diagnose * outputFeature.shape[0]
        diagnose = diagnose[:, np.newaxis]

        # 对抗
        domain_output = self.AdModel(outputFeature, alpha)
        #print(outputFeature.shape)
        result = self.Regress(outputFeature, gender, diagnose)
        result = result.squeeze(-1) #结果降一维
        attResult = self.AttentionModel(outputFeature)
        attResult = attResult.squeeze(-1)
        return result + _gender + _diagnose, domain_output, attResult

if __name__ == '__main__':
    config = vilbert.BertConfig.from_json_file("/home/ywz/DANN_myself/vilbertModel/config/bert_base_2layer_2conect.json")

    model = Net(config)
    Conv1dModel = ConvNet1d()

    total = sum([param.nelement() for param in model.parameters()]) 
    print("Number of parameter: %.2fM" % (total/1e6))
    model = model.state_dict()
    new_state_dict = OrderedDict()
    for k, v in model.items():
        if 'Transformer' in k  or 'TCN' in k or 'AdModel' in k or "Conv1dModel" in k:
            name = k
            new_state_dict[name] = v  
    for name in new_state_dict:
        print(name)
        #Conv1dModel.model1.1.running_mean
        # Conv1dModel.model1.1.running_var

    


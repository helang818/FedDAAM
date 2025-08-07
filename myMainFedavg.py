import argparse
import logging

logging.getLogger().setLevel(logging.NOTSET)

import os
import random
import socket
import sys
import traceback

import numpy as np
import psutil
import setproctitle
import torch
from mpi4py import MPI

from MyModel17 import Net as Net17
from MyModel14 import Net as Net14

from MyModel13 import Net as Net13

import vilbertModel.vilbert.vilbert as vilbert
from MyDataLoader2017 import MyDataLoader as MyDataLoader2017
from MyDataLoader2014N import MyDataLoader as MyDataLoader2014
from MyDataLoader2017 import AffectnetSampler as AffectnetSampler2017
from MyDataLoader2014N import AffectnetSampler as AffectnetSampler2014

from MyDataLoader2013 import MyDataLoader as MyDataLoader2013
from MyDataLoader2013 import AffectnetSampler as AffectnetSampler2013

from torch.utils.data import DataLoader

sys.path.append("/home/zhaojunnan/code/nfs/share/FedML")
sys.path.append("/home/zhaojunnan/code/nfs/share")
sys.path.append("/home/zhaojunnan/code/nfs/share/FedML/fedml_api/distributed")

# from fedml_api.distributed.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed
from FedML.fedml_api.distributed.fedavg.myFedAvgAPI import FedML_init, FedML_FedAvg_distributed
from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

batch=15

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=200,
                        help='how many round of communications we shoud use')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=1,
                        help='gpu_num_per_server')

    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str, default="mapping_config_5_3",
                        help='the key in gpu utilization file')

    parser.add_argument('--grpc_ipconfig_path', type=str, default="grpc_ipconfig.csv",
                        help='config table containing ipv4 address of grpc server')
    
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='config table containing ipv4 address of grpc server')
                        
    #parser.add_argument('--train_device', default=[2,2,2],help='决定训练所在gpu，分别代表服务端，客户端，客户端位置')
    
    args = parser.parse_args()
    return args

def create_model(process_id,device):
    config = vilbert.BertConfig.from_json_file("/home/zhaojunnan/code/nfs/share/FedML/fedml_experiments/distributed/fedavg/vilbertModel/config/bert_base_2layer_2conect.json")
    
    # 加载模型到GPU
    if torch.cuda.is_available():
        if process_id == 1:
            return Net17(config,device).cuda(device)
        elif process_id == 2:
            return Net14(config,device).cuda(device)
        elif process_id == 3:
            return Net13(config,device).cuda(device)
        elif process_id == 0:
            return Net14(config,device).cuda(device)
        else:
            print('process_id error when creat model!!!')
            return
    else:
        print("torch.cuda.is not available()")

def load_data(trainVideoPath, trainAudioPath, devVideoPath, devAudioPath, process_id, trainVideoPathMDN="", devVideoPathMDN=""):
    if process_id == 1 :
        trainSet = MyDataLoader2017(trainVideoPath, trainAudioPath, "train")
        sampler = AffectnetSampler2017(trainSet)
        trainLoader = DataLoader(trainSet, batch_size=batch, shuffle=True)
        devSet = MyDataLoader2017(devVideoPath, devAudioPath, "dev")
        devLoader = DataLoader(devSet, batch_size=batch, shuffle=False) 
        return trainLoader, devLoader
        
    elif process_id == 2:
        trainSet = MyDataLoader2014(trainVideoPath, trainVideoPathMDN, trainAudioPath, "train")
        sampler = AffectnetSampler2014(trainSet)
        trainLoader = DataLoader(trainSet, batch_size=batch, shuffle=True)
        devSet = MyDataLoader2014(devVideoPath, devVideoPathMDN, devAudioPath, "dev")
        devLoader = DataLoader(devSet, batch_size=batch, shuffle=False) 
        return trainLoader, devLoader
        
    elif process_id == 3:
        trainSet = MyDataLoader2013(trainVideoPath, trainVideoPathMDN, trainAudioPath, "train")
        sampler = AffectnetSampler2013(trainSet)
        trainLoader = DataLoader(trainSet, batch_size=batch, shuffle=True)
        devSet = MyDataLoader2013(devVideoPath, devVideoPathMDN, devAudioPath, "dev")
        devLoader = DataLoader(devSet, batch_size=batch, shuffle=False) 
        return trainLoader, devLoader
    else:
        print('process_id error when load data!!!')
        return
    

if __name__ == "__main__":
    logging.info("FedML start")
    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == 'darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()
    #worker_number=3
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # customize the process name
    str_process_name = "FedAvg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(level=logging.DEBUG,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))
    '''
    process ID = 1, host name = server2, AVEC2017
    process ID = 2, host name = jiang, AVEC2014
    process ID = 3, host name = jiang
    '''
        # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Please check "GPU_MAPPING.md" to see how to define the topology
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key)
    print('######',device)
    # load data
    trainVideoPath = ["/data1/zhaojunnan/new_FedML/2017/tcnFeature/train", "/data1/zhaojunnan/new_FedML/2014/tcnFeature/train","/data1/zhaojunnan/new_FedML/2014/images/train", "/data1/zhaojunnan/new_FedML/2013/tcnFeature/train","/data1/zhaojunnan/new_FedML/2013/images/train"]
    trainAudioPath = ["/data1/zhaojunnan/new_FedML/2017/audio_186/train", "/data1/zhaojunnan/new_FedML/2014/audio_186/train" , "/data1/zhaojunnan/new_FedML/2013/audio_186/train"]
    devVideoPath = ["/data1/zhaojunnan/new_FedML/2017/tcnFeature/dev","/data1/zhaojunnan/new_FedML/2014/tcnFeature/dev","/data1/zhaojunnan/new_FedML/2014/images/dev","/data1/zhaojunnan/new_FedML/2013/tcnFeature/dev","/data1/zhaojunnan/new_FedML/2013/images/dev"]
    devAudioPath = ["/data1/zhaojunnan/new_FedML/2017/audio_186/dev", "/data1/zhaojunnan/new_FedML/2014/audio_186/dev", "/data1/zhaojunnan/new_FedML/2013/audio_186/dev"]
    if process_id == 1:
        trainLoader, devLoader = load_data(trainVideoPath[0], trainAudioPath[0], devVideoPath[0], devAudioPath[0], process_id)
    if process_id == 2:
        trainLoader, devLoader = load_data(trainVideoPath[1], trainAudioPath[1], devVideoPath[1], devAudioPath[1], process_id, trainVideoPath[2], devVideoPath[2])
    if process_id == 3:
        trainLoader, devLoader = load_data(trainVideoPath[3], trainAudioPath[2], devVideoPath[3], devAudioPath[2], process_id, trainVideoPath[4], devVideoPath[4])
    if process_id == 0:
        trainLoader, devLoader = None, None

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(process_id,device)
    logging.info("create_model over:{}".format(process_id))

    if process_id != 0:
        train_data_num = len(trainLoader) 
    else:
        train_data_num = 0
    FedML_FedAvg_distributed(process_id, worker_number, device, comm,
                             model, train_data_num, trainLoader, devLoader, args)

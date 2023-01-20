import torch # 防止jittor发生动态库调用问题
import jittor as jt # jittor backend
from jittor import Var, models, nn
from jittor.dataset import Dataset
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import scipy.io as sio # for loading .mat file
import scipy.spatial as spa # for Delaunay triangulation
from sklearn.decomposition import PCA as PCAdimReduc
import itertools
import numpy as np
from PIL import Image
import pathlib
from MyDataset import MyDataset
from Benchmark import GMBenchmark
from GM_models import GMNet
from tqdm import tqdm

pygm.BACKEND = 'jittor' # set default backend for pygmtools
jt.flags.use_cuda = jt.has_cuda

classes = ["Car","Duck","Face","Motorbike","Winebottle"]

def getPaths(rootPath="data/WillowObject/WILLOW-ObjectClass"):
    rootPath = pathlib.Path(rootPath)
    categories = ["Car","Duck","Face","Motorbike","Winebottle"]
    trainImgPaths = []
    testImgPaths = []
    trainKptPaths = []
    testKptPaths = []
    
    for category in categories:
        category_image_paths = list(rootPath.glob(category+'/*.png'))
        category_image_paths = [str(path) for path in category_image_paths ]
        category_mat_paths = list(rootPath.glob(category+'/*.mat'))
        category_mat_paths = [str(path) for path in category_mat_paths ]
        len = category_image_paths.__len__()
        trainLen = (int)(len*0.8)
        
        trainImgPaths.append(category_image_paths[0:24])
        trainKptPaths.append(category_mat_paths[0:24])
        testImgPaths.append(category_image_paths[24:30])
        testKptPaths.append(category_mat_paths[24:30])
    return trainImgPaths,trainKptPaths,testImgPaths,testKptPaths

trainImgPaths,trainKptPaths,testImgPaths,testKptPaths = getPaths()
trainDataset = MyDataset(trainImgPaths,trainKptPaths,batch_size=3)
testDataset = MyDataset(testImgPaths,testKptPaths,batch_size=3)


max_epoch = 10

vgg16_cnn = models.vgg16_bn(True)
model = GMNet(vgg16_cnn)

optimizer = jt.optim.Adam(model.parameters(), lr=1e-4)


Benchmark = GMBenchmark(trainDataset,testDataset,model,classes,optimizer)

Benchmark.train(max_epoch)
Benchmark.eval()
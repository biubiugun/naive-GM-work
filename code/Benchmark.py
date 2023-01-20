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
from GM_models import GMNet
from tqdm import tqdm

pygm.BACKEND = 'jittor' # set default backend for pygmtools
jt.flags.use_cuda = jt.has_cuda


obj_resize = (256, 256)

train_class:int = 5


class GMBenchmark():
    def __init__(self,train_dataset:MyDataset,test_dataset:MyDataset,model:GMNet,categories,optim):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.categories = categories
        self.categoriesNum = len(categories)
        self.optim = optim
        
    
    def train(self,epochNum):
        print("----------train begin-------")
        bounds = [0,0,0,0,0]
        for epoch in range(epochNum):
            total_loss = []
            total_accu = []
            for categoryNumber in range(train_class):#self.categoriesNum
                if((bounds[(int)(categoryNumber)]>0.95) and (categoryNumber!=0) and (categoryNumber!=4)):
                    bounds[(int)(categoryNumber)] = 0
                    continue;
                self.train_dataset.setCategory((int)(categoryNumber))
                bound_accu = []
                for batch_idx,(jittor_img1,kpts1,A1,jittor_img2,kpts2,A2) in enumerate(tqdm(self.train_dataset)):
                    accu = 0.0
                    loss = jt.Var(0.0)
                    X = self.model(jittor_img1, jittor_img2, kpts1, kpts2,A1,A2)
                    for i in range(X.shape[0]):  
                        X_index = X[(int)(i)]
                        X_gt = jt.init.eye(X_index.shape[0])
                        loss += pygm.utils.permutation_loss(X_index, X_gt)
                        precision = (X_index * X_gt).sum() / X_index.sum()
                        accu+=((float)(precision))
                        
                    
                    loss = loss/self.train_dataset.batch_size    
                    accu = accu/self.train_dataset.batch_size
                    
                    bound_accu.append(accu)
                    
                    total_loss.append((float)(loss))
                    total_accu.append(accu)
                    
                    self.optim.backward(loss)
                    
                    self.optim.step()
                    self.optim.zero_grad()

                bounds[(int)(categoryNumber)] = np.mean(bound_accu)

                
                   
            print("Epoch{}: loss=={:.6f} accuracy=={:.6f}".format(epoch,np.mean(total_loss),np.mean(total_accu)))
            
        print("----------------train finish-----------")       
        
        

    def eval(self):
        print("----------eval---------")

        accus = []
        total_accus = []
        for i in range(train_class):
            accus.append([])
        for categoryNumber in range(train_class):#self.categoriesNum
            self.test_dataset.setCategory((int)(categoryNumber))
            # print(self.test_dataset.__len__())
            for batch_idx,(jittor_img1,kpts1,A1,jittor_img2,kpts2,A2) in enumerate(self.test_dataset):
                accu = 0.0
                loss = jt.Var(0.0)
                    
                X = self.model(jittor_img1, jittor_img2, kpts1, kpts2,A1,A2)
                for i in range(X.shape[0]):  
                    X_index = X[(int)(i)]
                    X_gt = jt.init.eye(X_index.shape[0])
                    loss += pygm.utils.permutation_loss(X_index, X_gt)
                    precision = (X_index * X_gt).sum() / X_index.sum()
                    accu+=((float)(precision))
                        
                    
                loss = loss/self.test_dataset.batch_size    
                accu = accu/self.test_dataset.batch_size    
                    
                # total_loss.append((float)(loss))
                accus[(int)(categoryNumber)].append(accu)
                total_accus.append(accu)
                   
                    
                
            # print(accus[1])
            # print(total_accus)      
        print("Car accuracy: {:.6f}\n Duck accuracy accuracy: {:.6f}\n Face accuracy accuracy: {:.6f}\n Motorbike accuracy accuracy: {:.6f}\n Winebottle accuracy: {:.6f}\n total accuracy: {:.6f}\n".
                format(np.mean(accus[0]),np.mean(accus[1]),np.mean(accus[2]),np.mean(accus[3]),np.mean(accus[4]),np.mean(total_accus)))
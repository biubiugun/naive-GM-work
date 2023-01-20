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
from tqdm import tqdm
pygm.BACKEND = 'jittor' # set default backend for pygmtools
jt.flags.use_cuda = jt.has_cuda


obj_resize = (256, 256)

def delaunay_triangulation(kpt):
    d = spa.Delaunay(kpt.numpy().transpose())
    A = jt.zeros((len(kpt[0]), len(kpt[0])))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A


class MyDataset(Dataset):
    def __init__(self,imgPaths,kptPaths,batch_size=1,shuffle=False):
        super(MyDataset,self).__init__()
        super(MyDataset,self).set_attrs(batch_size=batch_size,shuffle=shuffle)
        self.batch_size = batch_size
        self.lens = []
        self.nowLen = 0
        self.total_len = 0
        self.categories = ["Car","Duck","Face","Motorbike","Winebottle"]
        self.nowCategoty = 0
        self.kptPaths = kptPaths
        self.imgPaths = imgPaths
       
        for category_paths in imgPaths:
            self.lens.append(len(category_paths))
            self.total_len += len(category_paths)
        print(self.lens)
        
    
    def setCategory(self,category:int):
        self.nowCategoty = category
        self.nowLen = self.lens[category]
        
    def getImage(self,imgPath,kptPath):
        img = Image.open(imgPath)
        kpt = jt.Var(sio.loadmat(kptPath)['pts_coord'])
        oralImageSize0 = img.size[0]
        oralImageSize1 = img.size[1]
        kpt[0] = kpt[0] * obj_resize[0] / oralImageSize0
        kpt[1] = kpt[1] * obj_resize[1] / oralImageSize1
        A = (delaunay_triangulation(kpt))
        img = img.resize(obj_resize, resample=Image.BILINEAR)
        jittor_img = jt.Var(np.array(img, dtype=np.float32) / 256).permute(2, 0, 1).unsqueeze(0)
        jittor_img = jt.reshape(jittor_img,(3,256,256))
        return jittor_img,kpt,A
        
        
    
    
    def __getitem__(self,index:int):
        img,kpt,A = self.getImage(self.imgPaths[self.nowCategoty][index],self.kptPaths[self.nowCategoty][index])
        return jt.reshape(img,(1,3,256,256)),jt.reshape(kpt,(1,2,10)),jt.reshape(A,(1,10,10))
    
    def __iter__(self):
        index:int = 0
        while(index + self.batch_size*2<=self.nowLen):
            image1,kpt1,A1 = self[index]
            image2,kpt2,A2 = self[index+self.batch_size]
            for i in range(1,self.batch_size):
                img1_mid,kpt1_mid,A1_mid = self[(int)(index+i)]
                img2_mid,kpt2_mid,A2_mid = self[(int)(index+self.batch_size+i)]
                image1 = jt.concat((image1,img1_mid),dim=0)
                kpt1 = jt.concat((kpt1,kpt1_mid),dim=0)
                A1 = jt.concat((A1,A1_mid),dim=0)
                image2 = jt.concat((image2,img2_mid),dim=0)
                kpt2 = jt.concat((kpt2,kpt2_mid),dim=0)
                A2 = jt.concat((A2,A2_mid),dim=0)
            index = index+self.batch_size*2
            yield image1,kpt1,A1,image2,kpt2,A2
       
    def __len__(self):
        return self.nowLen
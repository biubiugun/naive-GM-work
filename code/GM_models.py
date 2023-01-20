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

def local_response_norm(input: Var, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0) -> Var:
    """
    jittor implementation of local_response_norm
    """
    dim = input.ndim
    assert dim >= 3

    if input.numel() == 0:
        return input

    div = input.multiply(input).unsqueeze(1)
    if dim == 3:
        div = nn.pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = nn.avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = nn.pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = nn.AvgPool3d((size, 1, 1), stride=1)(div).squeeze(1)
        div = div.view(sizes)
    div = div.multiply(alpha).add(k).pow(beta)
    return input / div


def l2norm(node_feat):
    return local_response_norm(
        node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)
    




class CNNNet(jt.nn.Module):
    def __init__(self, vgg16_module):
        super(CNNNet, self).__init__()
        # The naming of the layers follow ThinkMatch convention to load pretrained models.
        self.node_layers = jt.nn.Sequential(*[_ for _ in list(vgg16_module.features)[:31]])
        self.edge_layers = jt.nn.Sequential(*[_ for _ in list(vgg16_module.features)[31:38]])

    def execute(self, inp_img):
        feat_local = self.node_layers(inp_img)
        feat_global = self.edge_layers(feat_local)
        return feat_local, feat_global

class GMNet(jt.nn.Module):
    def __init__(self,vgg16_cnn):
        super(GMNet, self).__init__()
        self.gm_net = pygm.utils.get_network(pygm.pca_gm, pretrain=False) # fetch the network object
        self.cnn = CNNNet(vgg16_cnn)
         
        path = pygm.utils.download('vgg16_pca_voc_jittor.pt', 'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1qLxjcVq7X3brylxRJvELCbtCzfuXQ24J')
        self.cnn.load_state_dict(jt.load(path))
    
    def execute_old(self, img1, img2, kpts1, kpts2,A1,A2):
        # CNN feature extractor layers
        
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)
        feat1_local = l2norm(feat1_local)
        feat1_global = l2norm(feat1_global)
        feat2_local = l2norm(feat2_local)
        feat2_global = l2norm(feat2_global)

        # upsample feature map
        feat1_local_upsample = jt.nn.interpolate(feat1_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat1_global_upsample = jt.nn.interpolate(feat1_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat2_local_upsample = jt.nn.interpolate(feat2_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat2_global_upsample = jt.nn.interpolate(feat2_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat1_upsample = jt.concat((feat1_local_upsample, feat1_global_upsample), dim=1)
        feat2_upsample = jt.concat((feat2_local_upsample, feat2_global_upsample), dim=1)

        # assign node features
        rounded_kpts1 = jt.round(kpts1).long()
        rounded_kpts2 = jt.round(kpts2).long()
        node1 = feat1_upsample[0, :, rounded_kpts1[0,1], rounded_kpts1[0,0]].t()  # shape: NxC
        node2 = feat2_upsample[0, :, rounded_kpts2[0,1], rounded_kpts2[0,0]].t()  # shape: NxC

        node1 = jt.reshape(node1,(1,node1.shape[0],node1.shape[1]))
        node2 = jt.reshape(node2,(1,node2.shape[0],node2.shape[1]))
        X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net)
        return X

    def execute(self, img1, img2, kpts1, kpts2,A1,A2):
        # CNN feature extractor layers
        
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)
        feat1_local = l2norm(feat1_local)
        feat1_global = l2norm(feat1_global)
        feat2_local = l2norm(feat2_local)
        feat2_global = l2norm(feat2_global)

        # upsample feature map
        feat1_local_upsample = jt.nn.interpolate(feat1_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat1_global_upsample = jt.nn.interpolate(feat1_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat2_local_upsample = jt.nn.interpolate(feat2_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat2_global_upsample = jt.nn.interpolate(feat2_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat1_upsample = jt.concat((feat1_local_upsample, feat1_global_upsample), dim=1)
        feat2_upsample = jt.concat((feat2_local_upsample, feat2_global_upsample), dim=1)#batch,channel,n,n
        # assign node features
        rounded_kpts1 = jt.round(kpts1).long()
        rounded_kpts2 = jt.round(kpts2).long()
       
        batch_size = feat1_upsample.shape[0]
        node1 = feat1_upsample[0, :, rounded_kpts1[0,1], rounded_kpts1[0,0]].t()  # shape: NxC
        N = node1.shape[0]
        C = node1.shape[1]
        node1 = jt.reshape(node1,(1,N,C))
        node2 = feat2_upsample[0, :, rounded_kpts2[0,1], rounded_kpts2[0,0]].t()  # shape: NxC
        node2 = jt.reshape(node2,(1,N,C))
        for index in range(1,batch_size):
            node1_mid = feat1_upsample[(int)(index), :, rounded_kpts1[(int)(index),1], rounded_kpts1[(int)(index),0]].t()
            node2_mid = feat2_upsample[(int)(index), :, rounded_kpts2[(int)(index),1], rounded_kpts2[(int)(index),0]].t()
            node1_mid = jt.reshape(node1_mid,(1,N,C))
            node2_mid = jt.reshape(node2_mid,(1,N,C))
           
            node1 = jt.concat((node1,node1_mid),dim=0)
            node2 = jt.concat((node2,node2_mid),dim=0)

        X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused

        
        return X
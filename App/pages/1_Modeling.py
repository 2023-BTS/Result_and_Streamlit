import os
import cv2
import time
import natsort
import glob
from collections import Counter
from tqdm import tqdm
from PIL import Image

import streamlit as st
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from typing import Any, Callable, List, Optional, Type, Union

st.set_page_config(
    page_title='Modeling'
)
st.title('데이터 모델링')

state = {
    'data_channel': None, 
    'data_resize': None,
    'data_preprocessing': None,
    'model': None
}


################################################################################################################
############################################### 필요한 함수 정의 ################################################
################################################################################################################

# 0. 전처리
def BGR2RGB(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img

def Normalization_Color(image):
    normalize_img = cv2.normalize(BGR2RGB(image), None, 0, 255, cv2.NORM_MINMAX)
    return normalize_img

def Normalization_Gray(image):
    img = cv2.cvtColor(BGR2RGB(image), cv2.COLOR_RGB2GRAY)
    normalize_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return normalize_img

def HE_Color(image):
    img = cv2.cvtColor(BGR2RGB(image), cv2.COLOR_RGB2YCrCb) 
    img_planes = cv2.split(img) 
    img_planes_0 = cv2.equalizeHist(img_planes[0])
    merge_img = cv2.merge([img_planes_0, img_planes[1], img_planes[2]])
    he_img = cv2.cvtColor(merge_img, cv2.COLOR_YCrCb2RGB) 
    return he_img

def HE_Gray(image):
    img = cv2.cvtColor(BGR2RGB(image), cv2.COLOR_RGB2GRAY) 
    he_img = cv2.equalizeHist(img) 
    return he_img

def NHE_Color(image):
    img = cv2.cvtColor(Normalization_Color(image), cv2.COLOR_RGB2YCrCb)
    img_planes = cv2.split(img) 
    img_planes_0 = cv2.equalizeHist(img_planes[0])
    merge_img = cv2.merge([img_planes_0, img_planes[1], img_planes[2]])
    nhe_img = cv2.cvtColor(merge_img, cv2.COLOR_YCrCb2RGB)
    return nhe_img

def NHE_Gray(image):
    nhe_img = cv2.equalizeHist(Normalization_Gray(image))
    return nhe_img

# 0. ResNet18
def conv3x3(in_planes : int, out_planes : int, stride : int=1, groups : int =1, dilation : int=1) -> nn.Conv2d:
    '3x3 convolution with padding'
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, 
                     groups=groups, bias=False, dilation =dilation)

def conv1x1(in_planes : int, out_planes : int, stride : int=1) -> nn.Conv2d:
    '1x1 convolution'
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride = stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes:int,
        planes:int,
        stride:int=1,
        downsample: Optional[nn.Module]=None,
        groups:int=1,
        dilation:int=1,
        norm_layer: Optional[Callable[..., nn.Module]]=None
    )-> None:
        super(BasicBlock,self).__init__()
        
        #Normalization layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1= conv3x3(inplanes, planes, stride)
        self.bn1=norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2= norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x:Tensor)-> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        #downsampling이 필요한 경우 downsample 레이어를 block에 인자로 넣어주어야 함
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity #residual connection
        out = self.relu(out)
        return out
    
class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(
    self,
    block:Type[Union[BasicBlock, Bottleneck]],
    layers:List[int],
    num_classes : int=1000,
    zero_init_residual : bool=False,
    norm_layer: Optional[Callable[..., nn.Module]]=None
    )-> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer #batch norm layer
        self.inplanes = 64 #input shape
        self.dilation = 1 
        self.groups = 1
        
        #input block
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride =2, padding =3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride= 2, padding=1)
        
        #residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)
        
        #weight initalizaiton
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity = 'relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # zero-initialize the last BN in each residual branch
            # so that the residual branch starts with zero, and each residual block behaves like an identity
            # Ths improves the model by 0.2~0.3%
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)
            
    def _make_layer(self, block:Type[Union[BasicBlock, Bottleneck]],
                   planes:int, blocks:int, stride: int=1, dilate:bool=False)->nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        #downsampling 필요한 경우 downsample layer 생성
        if stride !=1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.dilation, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, dilation = self.dilation, 
                               norm_layer = norm_layer))

        return nn.Sequential(*layers)
    
    def forward(self, x:Tensor) -> Tensor:
        print('input shape:', x.shape)
        x = self.conv1(x)
        print('conv1 shape:', x.shape)
        x = self.bn1(x)
        print('bn1 shape:', x.shape)
        x = self.relu(x)
        print('relu shape:', x.shape)
        x = self.maxpool(x)
        print('maxpool shape:', x.shape)

        x = self.layer1(x)
        print('layer1 shape:', x.shape)
        x = self.layer2(x)
        print('layer2 shape:', x.shape)
        x = self.layer3(x)
        print('layer3 shape:', x.shape)
        x = self.layer4(x)
        print('layer4 shape:', x.shape)

        x = self.avgpool(x)
        print('avgpool shape:', x.shape)
        x = torch.flatten(x, 1)
        print('flatten shape:', x.shape)
        x = self.fc(x)
        print('fc shape:', x.shape)

        return x

# 0. Xception
def depthwise_separable_conv(input_dim, output_dim):

    depthwise_convolution = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim, bias=False)
    pointwise_convolution = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)

    model = nn.Sequential(
        depthwise_convolution,
        pointwise_convolution

    )

    return model

class entry_flow(nn.Module):
    def __init__(self):
        super(entry_flow, self).__init__()

        self.conv2d_init_1 = nn.Conv2d(in_channels = 3,
                                       out_channels = 32,
                                       kernel_size = 3,
                                       stride = 2,
                                      )

        self.conv2d_init_2 = nn.Conv2d(in_channels = 32,
                                       out_channels = 64,
                                       kernel_size = 3,
                                       stride = 1,
                                      )


        self.layer_1 = nn.Sequential(
            depthwise_separable_conv(input_dim = 64, output_dim = 128),
            nn.ReLU(),
            depthwise_separable_conv(input_dim = 128, output_dim = 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        )

        self.conv2d_1 = nn.Conv2d(in_channels = 64,
                                  out_channels = 128,
                                  kernel_size = 1,
                                  stride = 2
                                  )

        self.layer_2 = nn.Sequential(
            depthwise_separable_conv(input_dim = 128, output_dim = 256),
            nn.ReLU(),
            depthwise_separable_conv(input_dim = 256, output_dim = 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        )

        self.conv2d_2 = nn.Conv2d(in_channels = 128,
                                  out_channels = 256,
                                  kernel_size = 1,
                                  stride = 2
                                  )

        self.layer_3 = nn.Sequential(
            depthwise_separable_conv(input_dim = 256, output_dim = 728),
            nn.ReLU(),
            depthwise_separable_conv(input_dim = 728, output_dim = 728),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        )

        self.conv2d_3 = nn.Conv2d(in_channels = 256,
                                  out_channels = 728,
                                  kernel_size = 1,
                                  stride = 2
                                  )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_init_1(x)
        x = self.relu(x)
        x = self.conv2d_init_2(x)
        x = self.relu(x)

        output1_1 = self.layer_1(x)
        output1_2 = self.conv2d_1(x)
        output1_3 = output1_1 + output1_2


        output2_1 = self.layer_2(output1_3)
        output2_2 = self.conv2d_2(output1_3)
        output2_3 = output2_1 + output2_2


        output3_1 = self.layer_3(output2_3)
        output3_2 = self.conv2d_3(output2_3)
        output3_3 = output3_1 + output3_2
        y = output3_3

        return y


class middle_flow(nn.Module):
    def __init__(self):
        super(middle_flow, self).__init__()

        for i in range(7):
            layers = nn.Sequential(
                    nn.ReLU(),
                    depthwise_separable_conv(input_dim = 728, output_dim = 728),
                    nn.ReLU(),
                    depthwise_separable_conv(input_dim = 728, output_dim = 728),
                    nn.ReLU(),
                    depthwise_separable_conv(input_dim = 728, output_dim = 728)
                )
            self.add_module(f'layer_{i}', layers) # 각 레이어에 고유한 이름 부여

    def forward(self, x):
        for i in range(7):
            x_temp = getattr(self, f'layer_{i}')(x)
            x = x + x_temp
        return x



class exit_flow(nn.Module):
    def __init__(self, growth_rate=32):
        super(exit_flow, self).__init__()

        self.separable_network = nn.Sequential(
            nn.ReLU(),
            depthwise_separable_conv(input_dim = 728, output_dim = 728),
            nn.ReLU(),
            depthwise_separable_conv(input_dim = 728, output_dim = 1024),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2d_1 = nn.Conv2d(in_channels = 728,
                                  out_channels = 1024,
                                  kernel_size = 1,
                                  stride = 2
                                  )

        self.separable_conv_1 = depthwise_separable_conv(input_dim = 1024, output_dim = 1536)
        self.separable_conv_2 = depthwise_separable_conv(input_dim = 1536, output_dim = 2048)

        self.relu = nn.ReLU()
        self.avgpooling = nn.AdaptiveAvgPool2d((1))

        self.fc_layer = nn.Linear(2048, 2)

    def forward(self, x):
        output1_1 = self.separable_network(x)
        output1_2 = self.conv2d_1(x)
        output1_3 = output1_1 + output1_2

        y = self.separable_conv_1(output1_3)
        y = self.relu(y)
        y = self.separable_conv_2(y)
        y = self.relu(y)
        y = self.avgpooling(y)

        y = y.view(-1, 2048)
        y= self.fc_layer(y)


        return y

class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        self.entry_flow = entry_flow()
        self.middle_flow = middle_flow()
        self.exit_flow = exit_flow()



    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)

        return x

# 0. Few Shot Dataset
class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Predict query labels using labeled support images.
        '''
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores
    
class BaseDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(base_dir, '*.jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_name


class SupportDataset(BaseDataset):
    def __init__(self, base_dir, transform=None):
        super().__init__(base_dir, transform)
        self.normal_img_paths = sorted(glob.glob(os.path.join(base_dir, '0', '*.jpg')))
        self.abnormal_img_paths = sorted(glob.glob(os.path.join(base_dir, '1', '*.jpg')))
        
        self.img_paths = self.normal_img_paths + self.abnormal_img_paths
        self.img_labels = [0]*len(self.normal_img_paths) + [1]*len(self.abnormal_img_paths)

    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        label = self.img_labels[idx]
        return image, label


class QueryDataset(BaseDataset):
    def __init__(self, base_dir, transform=None):
        super().__init__(base_dir, transform)
    

def predict_with_support(support_loader, query_loader):
    all_predictions = []
    all_query_images = []  # 이미지 데이터도 저장하기 위한 리스트

    for support_images, support_labels in support_loader:
        support_images = support_images.cuda()
        support_labels = support_labels.cuda()

        with torch.no_grad():
            for query_images, image_names in query_loader:
                temp_preds = []
                
                # 각 이미지에 대한 예측을 5번 수행
                for _ in range(5):
                    outputs = model(support_images, support_labels, query_images.cuda())
                    _, preds = torch.max(outputs, 1)
                    temp_preds.append(preds)
                
                # 5번의 예측 중 가장 많이 예측된 값을 최종 예측값으로 선택
                for i, img_name in enumerate(image_names):
                    counter = Counter([temp[i].item() for temp in temp_preds])
                    most_common_pred = counter.most_common(1)[0][0]
                    all_predictions.append((img_name, most_common_pred))
                    all_query_images.append(query_images[i])  # 이미지 데이터 추가

    return all_predictions, all_query_images


################################################################################################################
############################################## Streamlit 구동 화면 ##############################################
################################################################################################################

# 1. 페이지 설명
# col0, _ = st.columns([9, 1])
# with col0:
#     with st.expander('활용 가이드'):
#         st.write('설명')
# st.write('')

# 2-1. 폴더명 입력
col1, col2 = st.columns([9, 1])
col_under, _ = st.columns([9, 1])
with col1:
    folder_name = st.text_input('Data Preview에서 입력한 폴더명을 입력한 후 제출 버튼을 클릭하세요.', key='folder_name_input')

# 2-2. 폴더명 제출
input_path = '/BTS2023/Streamlit_BTS/Data/input'
folder_name_path = os.path.join(input_path, folder_name)
with col2:
    st.write('')
    st.write('')
    if st.button('제출'):
        if not folder_name:
            col_under.error('폴더명을 입력하세요.')
        elif not os.path.exists(folder_name_path):
            col_under.error('동일한 폴더명이 존재하지 않습니다. 다시 입력하세요.')   
st.write('')

# 3. 데이터 채널 선택
col3, _ = st.columns([9, 1])
with col3:
    data_channel = st.selectbox(
        'Data Channel', 
        ('1', '3'))
st.write('')

# 4. 데이터 사이즈 선택
col4, _ = st.columns([9, 1])
with col4:
    data_resize = st.selectbox(
        'Data Resize', 
        ('256', '512'))
st.write('')

# 5. 데이터 전처리 선택
col5, _ = st.columns([9, 1])
with col5:
    data_preprocessing = st.selectbox(
        'Data Preprocessing (단일 선택)',
        ('None', 
         'Normalization', 
         'Histogram Equalization', 
         'Normalization & Histogram Equalization'))
st.write('')

# 6. 모델 선택 (채널, 사이즈, 전처리의 경우에 맞게 pth file 띄우기)
col6, col7 = st.columns([9, 1])
col_under, _ = st.columns([9, 1])

state['data_channel'] = data_channel
state['data_resize'] = data_resize
state['data_preprocessing'] = data_preprocessing

pth_path = '/BTS2023/Streamlit_BTS/Data/pth'
pth_list = natsort.natsorted(os.listdir(pth_path))

pth_final = []
for pth_file in pth_list:
    pth_file0 = pth_file.split('.')
    pth_file1 = pth_file0[0]
    pth_file2 = pth_file1.split('_')

    if state['data_channel'] == pth_file2[1] and state['data_resize'] == pth_file2[2] and state['data_preprocessing'] == pth_file2[3]:
        pth_final.append(f'{pth_file1}.pth')

with col6:                        
    model = st.selectbox(
        'Model (pth file)',
        tuple(pth_final)) # 변경 불가하도록 tuple로 처리

# 7. Run 및 모델링 진행률 (특정 경로에 이미지 및 csv file 저장)
with col7:
    st.write('')
    st.write('')
    if st.button('Run'):
        state['model'] = model
        
        # 7-1. 전처리 선택
        class CustomTransform:
            def __init__(self, data_preprocessing_=state['data_preprocessing'], data_channel_=state['data_channel']):
                self.data_preprocessing_ = data_preprocessing_
                self.data_channel_ = data_channel_
            
            def __call__(self, img):
                img = np.array(img)
                
                if self.data_channel_ == '1':
                    img = cv2.cvtColor(BGR2RGB(img), cv2.COLOR_RGB2GRAY)
                if self.data_preprocessing_ == 'Normalization':
                    img = Normalization_Color(img) if self.data_channel_ == '3' else Normalization_Gray(img)
                elif self.data_preprocessing_ == 'Histogram Equalization':
                    img = HE_Color(img) if self.data_channel_ == '3' else HE_Gray(img)
                elif self.data_preprocessing_ == 'Normalization & Histogram Equalization':
                    img = NHE_Color(img) if self.data_channel_ == '3' else NHE_Gray(img)
                
                img = img.astype(np.uint8)
                return Image.fromarray(img)
        
        # 7-2. CustomImageDataset 없어도 됨
        class CustomImageDataset(Dataset):
            def __init__(self, img_dir, transform=None):
                self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
                self.transform = transform

            def __len__(self):
                return len(self.img_paths)

            def __getitem__(self, idx):
                # img_path = os.path.join(self.img_dir, img_name)
                img_path = self.img_paths[idx]
                img_name = os.path.basename(img_path)
                image = Image.open(img_path).convert('RGB')

                if self.transform:
                    image = self.transform(image)

                return image, img_name  # 이미지와 함께 이미지 이름 반환

        # 7-3. 모델에 전처리 적용하기 위한 transforms
        def get_transforms(model_=state['model']):

            if '_1' in model_:
                data_channel = '1'
            elif '_3' in model_:
                data_channel = '3'
            
            if 'None' in model_:
                data_preprocessing = 'None'
            elif 'Normalization' in model_:
                data_preprocessing = 'Normalization'
            elif 'Histogram Equalization' in model_:
                data_preprocessing = 'Histogram Equalization'
            elif 'Normalization & Histogram Equalization' in model_:
                data_preprocessing = 'Normalization & Histogram Equalization'
            
            if '256' in model_:
                data_resize = (256, 256)
            elif '512' in model_:
                data_resize = (512, 512)
            
            transforms_list = [
                CustomTransform(data_preprocessing_=data_preprocessing, data_channel_=data_channel),
                transforms.Resize(data_resize),
                transforms.ToTensor()
            ]
            
            return transforms.Compose(transforms_list)
        
        # 7-4. 이미지 저장
        def save_to_csv(results_csv, folder_name=folder_name):
            save_csv_path = f'/BTS2023/Streamlit_BTS/Data/csv/{folder_name}'
            if not os.path.exists(save_csv_path):
                try:
                    os.makedirs(save_csv_path)
                except FileExistsError:
                    pass
            results_csv.to_csv(save_csv_path + f'/{folder_name}_result.csv', index=False)

        # 7-5. Model Cycle
        model_path = os.path.join(pth_path, state['model'])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_name = state['model']

        # 7-5-1. Xception
        if 'Xception' in model_path:
            model = Xception()
            model = torch.load(model_path)
            if '_1' in model_path:
                model.conv2d_init_1 = nn.Conv2d(in_channels = 1,
                                       out_channels = 32,
                                       kernel_size = 3,
                                       stride = 2,
                                      )

            all_preds = []
            all_indices = []

            data_transforms = {
                'test': get_transforms(state['model'])
            }
            test_dataset = BaseDataset(folder_name_path, transform=data_transforms['test'])
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


            # 진행률 바
            processed_images = 0
            num_images = len(test_loader.dataset)

            model.eval()
            model = model.to(device)
            with torch.no_grad():
                for i, (X, img_names) in enumerate(test_loader):
                    X = X.to(device)
                    output = model(X)
                    _, pred = torch.max(output, 1)

                    for j, prediction in enumerate(pred):
                        all_indices.append(img_names[j])
                        if prediction.item() == 1:
                            save_image(X[j].cpu(), os.path.join(f'/BTS2023/Streamlit_BTS/Data/output/{folder_name}/', img_names[j]))

                    all_preds.extend(pred.cpu().numpy())

                # 진행률 업데이트
                processed_images += len(test_loader)
                progress = num_images / num_images
                col_under.progress(progress)
                col_under.success('done')

            results = pd.DataFrame({
                'Index': all_indices,
                'Prediction': all_preds
            })
            results['Result'] = results.apply(lambda row: '불량' if row['Prediction'] == 1 else '정상', axis=1)

            with open(f'/BTS2023/Streamlit_BTS/Data/output/{folder_name}/{model_name}.txt', 'w') as file:
                pass
            save_to_csv(results, folder_name=folder_name)

        # 7-5-2. ResNet18
        elif 'ResNet18' in model_path:
            model_ResNet18 = ResNet(BasicBlock, [2,2,2,2])

            if '_1' in model_path:
                model_ResNet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            num_ftrs = model_ResNet18.fc.in_features
            model_ResNet18.fc = nn.Linear(num_ftrs, 2)
            model_ResNet18.load_state_dict(torch.load(model_path))

            all_preds = []
            all_indices = []

            data_transforms = {
                'test': get_transforms(state['model'])
            }

            test_dataset = BaseDataset(folder_name_path, transform=data_transforms['test'])
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


            # 진행률 바
            processed_images = 0
            num_images = len(test_loader.dataset)

            model_ResNet18.eval()
            model_ResNet18 = model_ResNet18.to(device)
            with torch.no_grad():
                for i, (X, img_names) in enumerate(test_loader):
                    X = X.to(device)
                    output = model_ResNet18(X)
                    _, pred = torch.max(output, 1)

                    for j, prediction in enumerate(pred):
                        all_indices.append(img_names[j])
                        if prediction.item() == 1:
                            save_image(X[j].cpu(), os.path.join(f'/BTS2023/Streamlit_BTS/Data/output/{folder_name}/', img_names[j]))

                    all_preds.extend(pred.cpu().numpy())

                # 진행률 업데이트
                processed_images += len(test_loader)
                progress = num_images / num_images
                col_under.progress(progress)
                col_under.success('done')

            results = pd.DataFrame({
                'Index': all_indices,
                'Prediction': all_preds
            })
            results['Result'] = results.apply(lambda row: '불량' if row['Prediction'] == 1 else '정상', axis=1)

            with open(f'/BTS2023/Streamlit_BTS/Data/output/{folder_name}/{model_name}.txt', 'w') as file:
                pass
            save_to_csv(results, folder_name=folder_name)

        # 7-5-3. Few Shot Learning
        elif 'Few Shot Learning' in model_path:
            convolutional_network = ResNet(BasicBlock, [2,2,2,2])
            if '_1' in model_path:
                convolutional_network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            num_ftrs = convolutional_network.fc.in_features
            convolutional_network.fc = nn.Linear(num_ftrs, 2)
            model = PrototypicalNetworks(convolutional_network).cuda()
            model.load_state_dict(torch.load(model_path))

            all_preds = []
            all_indices = []

            data_transforms = {
                'test': get_transforms(state['model'])
            }

            # jjaemni: 데이터셋 Streamlit_BTS 폴더로 옮겨야 함
            support_set = SupportDataset(base_dir='/BTS2023/byeonjun/BTS2023/few_shot/proto/train_data', transform=data_transforms['test'])
            query_set = QueryDataset(base_dir=folder_name_path, transform=data_transforms['test'])

            support_loader = DataLoader(support_set, batch_size=64, shuffle=True)
            query_loader = DataLoader(query_set, batch_size=64, shuffle=False)

            # 진행률 바
            processed_images = 0

            predictions, query_images = predict_with_support(support_loader, query_loader)

            for idx, (img_name, pred) in enumerate(predictions):

                all_indices.append(img_name)

                if pred == 1:
                    save_image(query_images[idx].cpu(), os.path.join(f'/BTS2023/Streamlit_BTS/Data/output/{folder_name}/', img_name))


                all_preds.append(pred)

            # 진행률 업데이트
            processed_images += len(predictions)
            progress = 1 / 1
            col_under.progress(progress)
            col_under.success('done')
                

            results = pd.DataFrame({
                'Index': all_indices,
                'Prediction': all_preds
            })
            results['Result'] = results.apply(lambda row: '불량' if row['Prediction'] == 1 else '정상', axis=1)

            with open(f'/BTS2023/Streamlit_BTS/Data/output/{folder_name}/{model_name}.txt', 'w') as file:
                pass
            save_to_csv(results, folder_name=folder_name)
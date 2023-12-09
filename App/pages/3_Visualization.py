import os 
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
from collections import Counter
from torch.utils.data import Dataset
import cv2


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# ----------------------------------ResNet18----------------------------------
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch import Tensor
from PIL import Image
from typing import Any, Callable, List, Optional, Type, Union

def conv3x3(in_planes : int, out_planes : int, stride : int=1, groups : int =1, dilation : int=1) -> nn.Conv2d:
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, 
                     groups=groups, bias=False, dilation =dilation)

def conv1x1(in_planes : int, out_planes : int, stride : int=1) -> nn.Conv2d:
    "1x1 convolution"
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
    
# ---------------------------------- ResNet18 ----------------------------------

# ---------------------------------- Xception ----------------------------------

import torch.nn as nn


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
            self.add_module(f"layer_{i}", layers) # 각 레이어에 고유한 이름 부여

    def forward(self, x):
        for i in range(7):
            x_temp = getattr(self, f"layer_{i}")(x)
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
    
# ---------------------------------- Xception ----------------------------------

# ---------------------------------- Fewshot ----------------------------------
# few shot dataset
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
        """
        Predict query labels using labeled support images.
        """
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

# -----------------
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

class CustomTransform:
    def __init__(self, data_preprocessing_, data_channel_):
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

def get_transforms(model_):

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

def preprocess_image(img):
    preprocessed_img = img.copy()[:, :, ::-1]  # BGR to RGB
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))  # HWC to CHW
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

import io
def overlay_image_on_mask(img, mask):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(mask, alpha=0.5, cmap='jet')
    ax.axis('off')
    
    # 결과를 BytesIO로 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # PIL 이미지로 변환
    overlayed_img = Image.open(buf)
    return overlayed_img

st.set_page_config(
    page_title='Visualization'
)
st.title("Grad-CAM Visualization")

# 폴더명 입력
col1, col2 = st.columns([9, 1])
col_under, _ = st.columns([9, 1])
with col1:
    folder_name = st.text_input('Data Preview에서 입력한 폴더명을 입력한 후 제출 버튼을 클릭하세요.', key='folder_name_input')

base_path = '/BTS2023/Streamlit_BTS/Data/input/'
folder_name_path = os.path.join(base_path, folder_name)

with col2:
    st.write('')
    st.write('')
    if st.button('제출'):
        if not os.path.exists(folder_name_path):
            col_under.error('동일한 폴더명이 존재하지 않습니다. 다시 입력하세요.')

# 미리 정의된 모델 목록을 얻기 위한 코드
model_dir = f'/BTS2023/Streamlit_BTS/Data/output/{folder_name}/'
model_files = [f[:-4] for f in os.listdir(model_dir) if f.endswith('.txt')]

selected_model_name = st.selectbox('Select or type a model name', model_files)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 사용자 모델 선택
if selected_model_name:
    if st.button("조회"):
        if 'Few Shot Learning' in selected_model_name:
            # 해당 모델 로드
            convolutional_network = ResNet(BasicBlock, [2,2,2,2])
            if '_1' in selected_model_name:
                convolutional_network.conv1 = nn.Conv2d(in_channels = 1,
                                       out_channels = 32,
                                       kernel_size = 3,
                                       stride = 2,
                                      )
                
            num_ftrs = convolutional_network.fc.in_features
            convolutional_network.fc = nn.Linear(num_ftrs, 2)
            model = PrototypicalNetworks(convolutional_network).cuda()
            model.load_state_dict(torch.load(f'/BTS2023/Streamlit_BTS/Data/pth/{selected_model_name}', map_location=next(convolutional_network.parameters()).device))

            model.eval()
            model = model.to(device)

            class GradCam(nn.Module):
                def __init__(self, model, module, layer):
                    super().__init__()
                    self.model = model
                    self.module = module
                    self.layer = layer
                    self.forward_result = None
                    self.backward_result = None
                    self.register_hooks()

                def register_hooks(self):
                    found_module = False
                    for module_n, module in self.model.named_modules():
                        if module_n == self.module:
                            found_module = True
                            found_layer = False
                            for layer_n, layer in module.named_children():
                                if layer_n == self.layer:
                                    found_layer = True
                                    layer.register_forward_hook(self.forward_hook)
                                    layer.register_backward_hook(self.backward_hook)
                                    break
                            if not found_layer:
                                raise ValueError(f"Layer '{self.layer}' not found in module '{self.module}'.")
                            break
                    if not found_module:
                        raise ValueError(f"Module '{self.module}' not found in the model.")

                def forward(self, input, target_index=None):
                    outs = self.model(input)
                    outs = outs.squeeze()

                    if target_index is None:
                        # target_index = outs.argmax()
                        target_index = 1

                    outs[target_index].backward(retain_graph=True)

                    a_k = torch.mean(self.backward_result, dim=(2, 3), keepdim=True)  # [batch_size, channels, 1, 1]
                    out = (a_k * self.forward_result).sum(dim=1)  # [batch_size, H, W]
                    out = F.relu(out)  # Apply ReLU
                    out = F.interpolate(out.unsqueeze(1), [256, 256], mode='bilinear', align_corners=False)  # Upsample to 256x256

                    return out.cpu().detach().squeeze().numpy()

                def forward_hook(self, _, input, output):
                    self.forward_result = output[0]

                def backward_hook(self, _, grad_input, grad_output):
                    self.backward_result = grad_output[0]


            grad_cam = GradCam(model=convolutional_network, module='layer3', layer='1')  # 계층 선택

            root = f'/BTS2023/Streamlit_BTS/Data/output/{folder_name}'
            img_list = sorted(glob.glob(os.path.join(root, '*.jpg')))
            transform = get_transforms(selected_model_name)

            for img_path in img_list:
                img = Image.open(img_path)
                input_tensor = transform(img).unsqueeze(0).to(device)

                # 이미지와 Grad-CAM 결과를 화면에 표시
                mask = grad_cam(input_tensor, None)

                # 원본 이미지가 512x512이면 256x256으로 크기 조정
                if img.size == (512, 512):
                    input_tensor = F.interpolate(input_tensor, size=(256, 256), mode='bilinear', align_corners=False)
                    img = transforms.ToPILImage()(input_tensor.squeeze().cpu())

                overlayed_img = overlay_image_on_mask(np.array(img), mask)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="원본: " + os.path.basename(img_path), use_column_width=True, channels='BGR')
                with col2:
                    st.image(overlayed_img, caption="Grad CAM: " + os.path.basename(img_path), use_column_width=True, channels='BGR')

        elif 'Xception' in selected_model_name:
            model = Xception()
            model = torch.load(f'/BTS2023/Streamlit_BTS/Data/pth/{selected_model_name}')

            if '_1' in selected_model_name:
                model.conv2d_init_1 = nn.Conv2d(in_channels = 1,
                                       out_channels = 32,
                                       kernel_size = 3,
                                       stride = 2,
                                      )
            model.eval()
            model = model.to(device)

            class GradCam(nn.Module):
                def __init__(self, model, module, layer):
                    super().__init__()
                    self.model = model
                    self.module = module
                    self.layer = layer
                    self.forward_result = None
                    self.backward_result = None
                    self.register_hooks()

                def register_hooks(self):
                    found_module = False
                    for module_n, module in self.model.named_modules():
                        if module_n == self.module:
                            found_module = True
                            found_layer = False
                            for layer_n, layer in module.named_children():
                                if layer_n == self.layer:
                                    found_layer = True
                                    layer.register_forward_hook(self.forward_hook)
                                    layer.register_backward_hook(self.backward_hook)
                                    break
                            if not found_layer:
                                raise ValueError(f"Layer '{self.layer}' not found in module '{self.module}'.")
                            break
                    if not found_module:
                        raise ValueError(f"Module '{self.module}' not found in the model.")

                def forward(self, input, target_index=None):
                    outs = self.model(input)
                    outs = outs.squeeze()

                    if target_index is None:
                        # target_index = outs.argmax()
                        target_index = 0

                    outs[target_index].backward(retain_graph=True)

                    a_k = torch.mean(self.backward_result, dim=(2, 3), keepdim=True)  # [batch_size, channels, 1, 1]
                    out = (a_k * self.forward_result).sum(dim=1)  # [batch_size, H, W]
                    out = F.relu(out)  # Apply ReLU
                    out = F.interpolate(out.unsqueeze(1), [256, 256], mode='bilinear', align_corners=False)  # Upsample to 256x256

                    return out.cpu().detach().squeeze().numpy()

                def forward_hook(self, _, input, output):
                    self.forward_result = output[0]

                def backward_hook(self, _, grad_input, grad_output):
                    self.backward_result = grad_output[0]


            grad_cam = GradCam(model=model, module='entry_flow', layer='layer_1')  # Xception 계층 선택

            root = f'/BTS2023/Streamlit_BTS/Data/output/{folder_name}'
            img_list = sorted(glob.glob(os.path.join(root, '*.jpg')))
            transform = get_transforms(selected_model_name)

            for img_path in img_list:
                img = Image.open(img_path)
                input_tensor = transform(img).unsqueeze(0).to(device)

                # 이미지와 Grad-CAM 결과를 화면에 표시
                mask = grad_cam(input_tensor, None)

                # 원본 이미지가 512x512이면 256x256으로 크기 조정
                if img.size == (512, 512):
                    input_tensor = F.interpolate(input_tensor, size=(256, 256), mode='bilinear', align_corners=False)
                    img = transforms.ToPILImage()(input_tensor.squeeze().cpu())

                overlayed_img = overlay_image_on_mask(np.array(img), mask)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="원본: " + os.path.basename(img_path), use_column_width=True, channels='BGR')
                with col2:
                    st.image(overlayed_img, caption="Grad CAM: " + os.path.basename(img_path), use_column_width=True, channels='BGR')

        elif 'ResNet18' in selected_model_name:
            model_ResNet18 = ResNet(BasicBlock, [2,2,2,2])
            
            if '_1' in selected_model_name:
                model_ResNet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            num_ftrs = model_ResNet18.fc.in_features
            model_ResNet18.fc = nn.Linear(num_ftrs, 2)
            model_ResNet18.load_state_dict(torch.load(f'/BTS2023/Streamlit_BTS/Data/pth/{selected_model_name}'))


            model_ResNet18.eval()
            model = model_ResNet18.to(device)

            class GradCam(nn.Module):
                def __init__(self, model, module, layer):
                    super().__init__()
                    self.model = model
                    self.module = module
                    self.layer = layer
                    self.forward_result = None
                    self.backward_result = None
                    self.register_hooks()

                def register_hooks(self):
                    found_module = False
                    for module_n, module in self.model.named_modules():
                        if module_n == self.module:
                            found_module = True
                            found_layer = False
                            for layer_n, layer in module.named_children():
                                if layer_n == self.layer:
                                    found_layer = True
                                    layer.register_forward_hook(self.forward_hook)
                                    layer.register_backward_hook(self.backward_hook)
                                    break
                            if not found_layer:
                                raise ValueError(f"Layer '{self.layer}' not found in module '{self.module}'.")
                            break
                    if not found_module:
                        raise ValueError(f"Module '{self.module}' not found in the model.")

                def forward(self, input, target_index=None):
                    outs = self.model(input)
                    outs = outs.squeeze()

                    if target_index is None:
                        # target_index = outs.argmax()
                        target_index = 1

                    outs[target_index].backward(retain_graph=True)

                    a_k = torch.mean(self.backward_result, dim=(2, 3), keepdim=True)  # [batch_size, channels, 1, 1]
                    out = (a_k * self.forward_result).sum(dim=1)  # [batch_size, H, W]
                    out = F.relu(out)  # Apply ReLU
                    out = F.interpolate(out.unsqueeze(1), [256, 256], mode='bilinear', align_corners=False)  # Upsample to 256x256

                    return out.cpu().detach().squeeze().numpy()

                def forward_hook(self, _, input, output):
                    self.forward_result = output[0]

                def backward_hook(self, _, grad_input, grad_output):
                    self.backward_result = grad_output[0]

            grad_cam = GradCam(model=model_ResNet18, module='layer3', layer='1')  # Xception 계층 선택

            root = f'/BTS2023/Streamlit_BTS/Data/output/{folder_name}'
            img_list = sorted(glob.glob(os.path.join(root, '*.jpg')))
            transform = get_transforms(selected_model_name)

            for img_path in img_list:
                img = Image.open(img_path)
                input_tensor = transform(img).unsqueeze(0).to(device)

                # 이미지와 Grad-CAM 결과를 화면에 표시
                mask = grad_cam(input_tensor, None)

                # 원본 이미지가 512x512이면 256x256으로 크기 조정
                if img.size == (512, 512):
                    input_tensor = F.interpolate(input_tensor, size=(256, 256), mode='bilinear', align_corners=False)
                    img = transforms.ToPILImage()(input_tensor.squeeze().cpu())

                overlayed_img = overlay_image_on_mask(np.array(img), mask)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="원본: " + os.path.basename(img_path), use_column_width=True, channels='BGR')
                with col2:
                    st.image(overlayed_img, caption="Grad CAM: " + os.path.basename(img_path), use_column_width=True, channels='BGR')
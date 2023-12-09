import streamlit as st
import zipfile
import os
import cv2
import time
import natsort
import random

st.set_page_config(
    page_title='Data Preview'
)
st.title('데이터 전처리 확인')

state = {
    'data_channel': None, 
    'data_preprocessing': None
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


################################################################################################################
############################################## Streamlit 구동 화면 ##############################################
################################################################################################################

# 1. 페이지 설명
col7, col8 = st.columns([9, 1])
# with col7:
#     with st.expander('활용 가이드'):
#         st.write('설명')
# st.write('')

# 2-1. 폴더명 입력
col0, col1 = st.columns([9, 1])
col_under, _ = st.columns([9, 1])
with col0:
    folder_name = st.text_input('원하는 폴더명을 입력한 후 제출 버튼을 클릭하세요. 입력한 폴더명으로 경로가 생성됩니다.', key='folder_name_input')

# 2-2. 폴더명 제출 및 생성
def make_dir(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except FileExistsError:
            pass

input_path = '/BTS2023/Streamlit_BTS/Data/input'
make_dir(input_path)
output_path = '/BTS2023/Streamlit_BTS/Data/output'
make_dir(output_path)
csv_path = '/BTS2023/Streamlit_BTS/Data/csv'
make_dir(csv_path)

folder_name_path = os.path.join(input_path, folder_name)
output_dataset_path = os.path.join(output_path, folder_name)
csv_dataset_path = os.path.join(csv_path, folder_name)

with col1:
    st.write('')
    st.write('')
    if st.button('제출'):
        if not folder_name:
            col_under.error('폴더명을 입력하세요.')
        elif os.path.exists(folder_name_path):
            col_under.error('동일한 폴더명이 존재합니다. 다시 입력하세요.')
        else:
            make_dir(folder_name_path)
            make_dir(output_dataset_path)
            make_dir(csv_dataset_path)
st.write('')

# 3-1. Zipfile 업로드
col2, col3 = st.columns([9, 1])
with col2:
    uploaded_file = st.file_uploader('Zipfile을 업로드하세요. 업로드가 완료되면 화면의 파일은 삭제가 가능합니다.', type=['zip'])

    # 3-2. Zipfile 저장
    uploaded_path = f'{input_path}/zipfile.zip'
    if uploaded_file is not None:
        with open(uploaded_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        # 3-3. Zipfile 압축 해제
        my_bar = st.empty()
        with zipfile.ZipFile(uploaded_path, 'r') as zip:
            file_list = zip.namelist()
            for i, file in enumerate(file_list):
                zip.extract(file, folder_name_path)
                progress_text = f'압축 해제 중... ({i + 1}/{len(file_list)})'
                my_bar.text(progress_text)
                time.sleep(0.05)
            st.success('done')
        os.remove(uploaded_path)
st.write('')

# 4. 데이터 채널 선택
col4, col5 = st.columns([9, 1])
with col4:
    data_channel = st.selectbox(
        'Data Channel', 
        ('1', '3'))
st.write('')

# 5. 데이터 전처리 선택
col5, col6 = st.columns([9, 1])
col9, col10, col11 = st.columns([4.5, 4.5, 1])
with col5:
    data_preprocessing = st.selectbox(
        'Data Preprocessing (단일 선택)',
        ('None', 
         'Normalization', 
         'Histogram Equalization', 
         'Normalization & Histogram Equalization'))
    
# 6-1. 이미지에 채널, 전처리 순으로 적용
def final_check():
    image_list = natsort.natsorted(os.listdir(folder_name_path))
    random_image_name = random.choice(image_list)
    random_image_path = os.path.join(folder_name_path, random_image_name)
    random_image_read = cv2.imread(random_image_path)
    image_cp = None

    if data_channel == '1':
        if data_preprocessing == 'None':
            image_cp = cv2.cvtColor(BGR2RGB(random_image_read), cv2.COLOR_RGB2GRAY)
        elif data_preprocessing == 'Normalization':
            image_cp = Normalization_Gray(random_image_read)
        elif data_preprocessing == 'Histogram Equalization':
            image_cp = HE_Gray(random_image_read)
        elif data_preprocessing == 'Normalization & Histogram Equalization':
            image_cp = NHE_Gray(random_image_read)

    elif data_channel == '3':
        if data_preprocessing == 'None':
            image_cp = BGR2RGB(random_image_read)
        elif data_preprocessing == 'Normalization':
            image_cp = Normalization_Color(random_image_read)
        elif data_preprocessing == 'Histogram Equalization':
            image_cp = HE_Color(random_image_read)
        elif data_preprocessing == 'Normalization & Histogram Equalization':
            image_cp = NHE_Color(random_image_read)

    if image_cp is not None:
        col9.image(random_image_read, caption=f'Original: {random_image_name}')
        col10.image(image_cp, caption=f'Transform: {random_image_name}')

# 6-2. 데이터 채널 및 전처리 적용
with col6:
    st.write('')
    st.write('')
    if st.button('조회', on_click=final_check):
        state['data_channel'] = data_channel
        state['data_preprocessing'] = data_preprocessing
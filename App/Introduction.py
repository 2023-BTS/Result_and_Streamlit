import streamlit as st

st.set_page_config(
    page_title='Welcome to BBC',
    page_icon='😊'
)

st.title('유리 진공관 램프 표면 결함 탐지')
st.subheader('*:red[B]est :red[B]usiness :red[C]rew*')
st.caption('BTS Member. 박시현 변준형 윤기창 이진 한은규 홍재민')
st.caption('👉 [GitHub](https://github.com/HongJaeMin)')
st.markdown('---')

st.subheader('프로젝트 기간: 2023.04. ~ 2023.12.')
st.markdown(
    '''
    0. 연구 주제
        - AI 기반 진공관 램프 결함탐지 및 제조공정 최적화
    '''
)
st.markdown(
    '''
    1. 연구 개요
        - 연구 배경
            - 진공관 램프 제조업인 선재 하이테크에서는 현재 50%라는 높은 불량률로 인해 약 2억 4천만 원의 손실을 겪고 있음. 상당한 손실을 개선하고자, 본 연구는 AI를 활용하여 진공관 램프의 표면 결함을 효과적으로 감지함으로써 제조 과정에서의 손실을 최소화하고 안정적인 품질의 제품을 제공하고자 함.
        - 개요
            - 현재 육안으로 검사하는 방식을 대체할 새로운 알고리즘을 개발함.
            - 제조업체 내에서 손쉽게 활용할 수 있는 웹 애플리케이션을 구축함.
        - 목표
            - 진공관 램프 표면 결함을 탐지하기 위한 효과적인 AI 기반 알고리즘 개발을 목표로 함.
            - 사용자 친화적인 웹 애플리케이션을 제작하여 사용 접근성 향상을 목표로 함.
    '''
)
st.markdown(
    '''
    2. 연구활동 주요 내용
        - 전처리
            - Data Channel, Image Resize, Normalization, Histogram Equalization을 사용함.
                - Data Channel : `1`, `3`
                - Image Resize : `256`, `512`
                - Data Preprocessing : `Normalization`, `Histogram Equalization`, `Normalization & Histogram Equalization`, `None`
        - 모델
            - ResNet18
            - Few Shot Learning
            - Xception
    '''
)
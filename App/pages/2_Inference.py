import streamlit as st 
import pandas as pd
from PIL import Image
import os

st.set_page_config(
    page_title='Inference'
)
st.title('결과 확인')

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

# CSV 파일 읽기와 다운로드 버튼
csv_file_path = f'/BTS2023/Streamlit_BTS/Data/csv/{folder_name}/{folder_name}_result.csv'
df_faulty = None  # df_faulty 초기화

if folder_name:  # 폴더명이 입력된 경우
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path, encoding="utf-8-sig")  # utf-8-sig로 읽기

        df_faulty = df[df['Prediction'] == 1]

        if st.button("CSV 파일 조회"):
            st.write(df.head(10))

        csv_data = df.to_csv(index=False, encoding="utf-8-sig")  # utf-8-sig로 저장
        st.download_button(
            label="CSV 전체 파일 다운로드",
            data=csv_data.encode("utf-8-sig"),
            file_name=f"{folder_name}_result.csv",
            mime="text/csv",
        )
    else:
        st.write("CSV 파일이 존재하지 않습니다.")
else:  # 폴더명이 입력되지 않은 경우
    st.write("폴더명을 입력해주세요.")


# 불량 이미지 보기
img_folder = f'/BTS2023/Streamlit_BTS/Data/output/{folder_name}'
max_cols_per_row = 5  # 한 행당 최대 열의 개수

if df_faulty is not None:
    faulty_images = df_faulty['Index'].tolist()

    if st.button("불량 이미지 조회"):
        if os.path.exists(img_folder) and faulty_images:
            for i in range(0, len(faulty_images), max_cols_per_row):
                images = faulty_images[i:i+max_cols_per_row]
                cols = st.columns(max_cols_per_row)  # 항상 동일한 수의 열을 생성

                for col, img_name in zip(cols, images):
                    img_path = os.path.join(img_folder, img_name)
                    img = Image.open(img_path)
                    col.image(img, caption=img_name, use_column_width=True)

                # 남은 열에 대해 빈 공간 채우기
                for _ in range(len(images), max_cols_per_row):
                    cols[_].write("")  # 빈 공간 생성
        else:
            st.write("불량 이미지가 존재하지 않습니다.")

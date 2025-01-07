import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch

from PIL import Image
from tsai.inference import load_learner
from sklearn.metrics import classification_report

from Hack_preprocess import *
from Hack_modeling import *

# font Setting
plt.rc('font', family='Malgun Gothic')
# Minus
matplotlib.rcParams['axes.unicode_minus'] = False

# def add_result(df):
#     result_mapping = {
#         '그루브깊이': ['그루브깊이1번_Result', '그루브깊이2번_Result', '그루브깊이3번_Result', '그루브깊이4번_Result', '그루브깊이5번_Result'],
#         '위치도': ['위치도1번_Result', '위치도2번_Result', '위치도3번_Result', '위치도4번_Result', '위치도5번_Result'],
#         '진원도': ['진원도1번_Result', '진원도2번_Result', '진원도3번_Result', '진원도4번_Result', '진원도5번_Result'],
#         '그루브경': ['그루브경1번_Result', '그루브경2번_Result', '그루브경3번_Result', '그루브경4번_Result', '그루브경5번_Result']
#     }

#     # ***_Result 열 4개 추가
#     for new_col, related_cols in result_mapping.items():
#         df[f'{new_col}_Result'] = df[related_cols].apply(lambda x: 0 if (x == 0).any() else 1, axis=1)
    
#     return df

# # 3. 길이 조정 함수 정의
# def adjust_length(data, target_len):
#     current_len = len(data)
#     if current_len < target_len:
#         # 패딩: 부족한 부분을 0으로 채움
#         return np.pad(data, ((0, target_len - current_len), (0, 0)), constant_values=0)
#     elif current_len > target_len:
#         # 잘라내기: 앞부분만 사용
#         return data[:target_len]
#     else:
#         # 이미 target_len과 같음
#         return data
    
# 전체 화면 설정
st.set_page_config(layout="wide")

# 상단에 이미지 추가
# image = Image.open("../plot/image.png")  # 업로드한 이미지 경로
# st.image(image, use_container_width=True)

st.markdown(
    "**FACTORY HACK KOREA 2025**"
)
# 앱 제목 및 설명
st.title("제조 데이터 불량 예측")
st.write("업로드한 제조 데이터를 분석하고, 불량 여부를 예측합니다.")

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    # 데이터 로드
    data = pd.read_csv(uploaded_file)
    data = add_result(data)
    # SerialNo 및 Result 컬럼 확인
    if 'SerialNo' not in data.columns:
        st.error("업로드한 파일에 'SerialNo' 열이 포함되어야 합니다.")
    else:
        st.success("파일이 성공적으로 업로드되었습니다!")

        # 1. 데이터 시각화
        feature_columns = [
            "ActF", "SpindleSpeed", "ModalT_x", "servoload_x", "servoload_z",
            "servocurrent_x", "servocurrent_z", "SpindleLoad"
        ]

    # 2열 레이아웃 설정
    col1, col2 = st.columns(2)

    # 1열: SerialNo 선택 및 시각화
    with col1:
        st.subheader("📈주요 변수 시각화")
        serial_numbers = data['SerialNo'].unique()
        selected_serial = st.selectbox("시각화 할 시리얼번호를 선택하세요", serial_numbers)

        if selected_serial:
            serial_data = data[data['SerialNo'] == selected_serial]

            fig = plot_features(serial_data)
            st.pyplot(fig)

    # 2열: 불량 여부 예측
    with col2:
        st.subheader("🔍불량 예측")
        st.write("공정 파트 별로 불량 확률을 예측합니다.")
        if st.button("불량 예측 실행"):
            try:
                result_df = inference(data)
                #st.write("예측 결과:")
                #st.dataframe(result_df)

                # Streamlit 내장 Bar Chart로 불량 확률 시각화
                st.subheader("예측 결과")
                fig = visualize_result(result_df)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"예측에 실패했습니다: {e}")

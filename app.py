import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from Hack_preprocess import *
from Hack_modeling import *

# font Setting
plt.rc('font', family='Malgun Gothic')
# Minus
matplotlib.rcParams['axes.unicode_minus'] = False

url = 'https://raw.githubusercontent.com/Hasaero/factory_hack_2025/master/font/malgun.ttf'
response = requests.get(url)

# 현재 디렉토리에 ttf 파일을 저장합니다.
with open('malgun.ttf', 'wb') as out_file:
    out_file.write(response.content)

# 이제 파일은 로컬 파일 시스템에 저장되어 있으므로 ft2font.FT2Font에서 사용할 수 있습니다.
font_path = os.path.abspath('malgun.ttf.ttf')
fm.fontManager.addfont(font_path)

# 위 코드는 캐시된 FontManager를 무시하고 새로운 것을 불러오도록 설정합니다.
fm._load_fontmanager(try_read_cache=False)

# 이제 'BM Dohyeon' 폰트를 사용할 수 있게 됐습니다.
plt.rc('font', family='Malgun Gothic')

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

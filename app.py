import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from Hack_preprocess import *
from Hack_modeling import *
from pathlib import Path

# 경로를 Path로 정의
font_path = Path('malgun.ttf')  # 운영 체제에 따라 적절한 경로 타입으로 설정

# Minus
matplotlib.rcParams['axes.unicode_minus'] = False

# 폰트 다운로드 및 캐싱
@st.cache_data
def download_and_set_font():
    url = 'https://raw.githubusercontent.com/Hasaero/factory_hack_2025/master/font/malgun.ttf'
    response = requests.get(url)

    # Streamlit 캐싱 디렉토리에 폰트 저장
    font_dir = Path(st.__path__[0]) / "static" / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    font_path = font_dir / 'malgun.ttf'

    # 폰트 저장
    with font_path.open('wb') as out_file:
        out_file.write(response.content)

    # matplotlib에 폰트 추가
    fm.fontManager.addfont(str(font_path))
    fm._load_fontmanager(try_read_cache=False)

    return str(font_path)

# 폰트 설정
font_path = download_and_set_font()
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
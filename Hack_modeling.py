#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random

from tsai.basics import *
from tsai.all import *

from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return


# 3. 길이 조정 함수 정의
def adjust_length(data, target_len):
    current_len = len(data)
    if current_len < target_len:
        # 패딩: 부족한 부분을 0으로 채움
        return np.pad(data, ((0, target_len - current_len), (0, 0)), constant_values=0)
    elif current_len > target_len:
        # 잘라내기: 앞부분만 사용
        return data[:target_len]
    else:
        # 이미 target_len과 같음
        return data


class RecallForZero(Metric):
    def reset(self):
        self.y_true = []
        self.y_pred = []

    def accumulate(self, learn):
        preds = learn.pred.argmax(dim=-1).cpu().numpy()
        targets = learn.y.cpu().numpy()
        self.y_true.extend(targets)
        self.y_pred.extend(preds)

    @property
    def value(self):
        return recall_score(self.y_true, self.y_pred, labels=[0], average=None)[0]
    

class PrecisionForZero(Metric):
    def reset(self):
        self.y_true = []
        self.y_pred = []

    def accumulate(self, learn):
        preds = learn.pred.argmax(dim=-1).cpu().numpy()
        targets = learn.y.cpu().numpy()
        self.y_true.extend(targets)
        self.y_pred.extend(preds)

    @property
    def value(self):
        return precision_score(self.y_true, self.y_pred, labels=[0], average=None)[0]
    

class F1ForZero(Metric):
    def reset(self):
        self.y_true = []
        self.y_pred = []

    def accumulate(self, learn):
        preds = learn.pred.argmax(dim=-1).cpu().numpy()
        targets = learn.y.cpu().numpy()
        self.y_true.extend(targets)
        self.y_pred.extend(preds)

    @property
    def value(self):
        return f1_score(self.y_true, self.y_pred, labels=[0], average=None)[0]



def train(raw_df):

    target_columns = [ 'Result', '내경_Result', '그루브깊이_Result', '위치도_Result', '진원도_Result', '그루브경_Result',]

    #target_columns=['진원도_Result']
    # 사용할 피처와 그룹화할 컬럼 정의
    feature_columns = ['ActF', 'SpindleSpeed', 'ModalT_x', 'servoload_x', 
                       'servoload_z', 'servocurrent_x', 'servocurrent_z', 'SpindleLoad']

    grouped = raw_df.groupby('SerialNo')  # SerialNo를 기준으로 그룹화

    # 2. SerialNo별 시계열 길이 계산
    series_lengths = grouped.size()  # SerialNo별 시계열 길이
    median_length = int(series_lengths.median())  # 중앙값 계산

    # 4. SerialNo별 데이터 길이 조정
    X = np.stack([
        adjust_length(group[feature_columns].values, median_length)
        for _, group in grouped
    ])

    # Train/Test Split
    train_idx = raw_df[raw_df['ReceivedDateTime'] < pd.to_datetime('2023-06-01')].groupby('SerialNo').ngroup().unique()  # 학습 데이터
    test_idx = raw_df[raw_df['ReceivedDateTime'] >= pd.to_datetime('2023-06-01')].groupby('SerialNo').ngroup().unique()  # 테스트 데이터
    splits = (list(train_idx), list(train_idx[-1] + test_idx + 1))

    for target in target_columns:
        print(f"#### {target} ####")
        # y값 생성 (SerialNo별 Result 값의 대표값 사용)
        y = grouped[target].first().values

        serials = grouped['SerialNo'].first().values  # SerialNo 리스트

        # 2. 모델 학습
        tfms = [None, TSClassification()]  # 데이터 변환
        batch_tfms = TSStandardize(by_sample=True)  # 표준화

        # 다변량 시계열 분류 모델 생성
        mv_clf = TSClassifier(X, y, splits=splits, path='models', 
                              arch="InceptionTimePlus", tfms=tfms, 
                              batch_tfms=batch_tfms, metrics=accuracy, cbs=ShowGraph())

        mv_clf.fit_one_cycle(5, 1e-2)  # 10 에포크 동안 학습
        mv_clf.export(f"{target}_inception.pkl")  # 모델 저장

        # 3. 모델 로드 및 예측
        mv_clf = load_learner(f"models/{target}_inception.pkl")  # 모델 로드
        X_test = X[splits[1]]  # 테스트 데이터
        y_test = y[splits[1]]  # 테스트 타겟값

        probas, actual, preds = mv_clf.get_X_preds(X_test, y_test)  # 확률, 실제값, 예측값

        # 4. 결과 출력
        test_serials = serials[splits[1]]  # 테스트 데이터에 해당하는 SerialNo
        results = pd.DataFrame({
            'SerialNo': test_serials,
            'Actual': actual,
            'Predicted': preds
        })

        # Actual과 Predicted 값을 가져오기
        actual = results['Actual']  # 이미 int 타입
        predicted = results['Predicted'].astype(int)  # 문자열을 정수형으로 변환

        # Classification Report 출력
        print(target)
        report = classification_report(actual, predicted, digits=4)
        print("Classification Report:")
        print(report)
    
    return


def inference(demo_df):

    target_columns = ['종합_Result', '내경_Result', '그루브깊이_Result', '위치도_Result', '진원도_Result', '그루브경_Result']
    # 사용할 피처와 그룹화할 컬럼 정의
    feature_columns = ['ActF', 'SpindleSpeed', 'ModalT_x', 'servoload_x', 
                       'servoload_z', 'servocurrent_x', 'servocurrent_z', 'SpindleLoad']
    grouped = demo_df.groupby('SerialNo')  # SerialNo를 기준으로 그룹화
    # 2. SerialNo별 시계열 길이 계산
    series_lengths = grouped.size()  # SerialNo별 시계열 길이
    median_length = 260  # 중앙값 계산

    # 4. SerialNo별 데이터 길이 조정
    X = np.stack([
        adjust_length(group[feature_columns].values, median_length) for _, group in grouped])

    result_df = pd.DataFrame()

    for target in target_columns:
        test_idx = demo_df[demo_df['ReceivedDateTime'] >= '2023-01'].groupby('SerialNo').ngroup().unique()  # 테스트 데이터 -> 전체 사용
        # 어차피 train demo_df 안 씀
        splits = (0, list(test_idx))
        # y값 생성 (SerialNo별 Result 값의 대표값 사용)
        y = grouped[target].first().values
        serials = grouped['SerialNo'].first().values  # SerialNo 리스트
        # 3. 모델 로드 및 예측
        mv_clf = load_learner(f"models/{target}_inception.pkl")  # 모델 로드
        X_test = X[splits[1]]  # 테스트 데이터
        y_test = y[splits[1]]  # 테스트 타겟값
        probas, _, preds = mv_clf.get_X_preds(X_test, y_test)  # 확률, 실제값, 예측값

        result_df = pd.concat([result_df, pd.DataFrame({'결과': [target], '불량일 확률': str(round(float(probas[0][0]) * 100, 2)) +'%'})])

    result_df["결과"] = result_df["결과"].str.split("_Result").str[0]

    result_df = result_df.set_index(result_df["결과"]).drop(columns=['결과'])

    return result_df

def visualize_result(result_df):

    # 서브플롯 생성
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 각 결과에 대해 파이 차트 그리기
    for i, ax in enumerate(axes):
        result = result_df.iloc[i]
        labels = ['불량', '정상']
        sizes = [result['불량일 확률'], 100 - result['불량일 확률']]  # 불량일 확률, 정상 확률
        colors = ['red', 'skyblue']

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title(result['결과'])

    # 전체 레이아웃 조정 및 출력
    plt.tight_layout()
    plt.show()
    
    return fig

def visualize_result(result_df):

    # '% 제거 및 float로 변환
    result_df['불량일 확률'] = result_df['불량일 확률'].str.rstrip('%').astype(float)
    result_df.reset_index(inplace=True)
    
    # 비즈니스 대시보드 스타일로 파이 차트 생성 (불량/정상 색상 변경)
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes = axes.flatten()

    # 각 결과에 대해 파이 차트 그리기
    for i, ax in enumerate(axes):
        result = result_df.iloc[i]
        labels = ['불량', '정상']
        sizes = [result['불량일 확률'], 100 - result['불량일 확률']]  # 불량일 확률, 정상 확률
        colors = ['#FF6F61', '#6EC4E8']  # 비즈니스 친화적인 색상 (빨강, 하늘색)

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 16}
        )

        # 웨지 스타일
        for wedge in wedges:
            wedge.set_edgecolor('white')
            wedge.set_linewidth(1.5)

        # 텍스트 스타일
        for j, text in enumerate(texts):
            if labels[j] == '불량 확률':
                text.set_color('#FF6F61')  # 빨간색
            elif labels[j] == '정상 확률':
                text.set_color('#6EC4E8')  # 하늘색

        for autotext in autotexts:
            autotext.set_color('#2E4053')  # 더 진한 색상
            autotext.set_weight('bold')  # 텍스트를 볼드 처리
            autotext.set_fontsize(16)  # 텍스트 크기 확대

        # 차트 제목
        ax.set_title(result['결과'], fontsize=20, color='#2E4053', weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=2.0, w_pad=2.5, h_pad=2.0)
    
    return fig

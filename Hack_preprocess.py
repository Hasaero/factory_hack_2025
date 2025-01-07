#!/usr/bin/env python
# coding: utf-8

# DataFrame
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib

def add_result(df):
    result_mapping = {
        '그루브깊이': ['그루브깊이1번_Result', '그루브깊이2번_Result', '그루브깊이3번_Result', '그루브깊이4번_Result', '그루브깊이5번_Result'],
        '위치도': ['위치도1번_Result', '위치도2번_Result', '위치도3번_Result', '위치도4번_Result', '위치도5번_Result'],
        '진원도': ['진원도1번_Result', '진원도2번_Result', '진원도3번_Result', '진원도4번_Result', '진원도5번_Result'],
        '그루브경': ['그루브경1번_Result', '그루브경2번_Result', '그루브경3번_Result', '그루브경4번_Result', '그루브경5번_Result']
    }

    # ***_Result 열 4개 추가
    for new_col, related_cols in result_mapping.items():
        df[f'{new_col}_Result'] = df[related_cols].apply(lambda x: 0 if (x == 0).any() else 1, axis=1)
    
    return df


def plot_features(serial_data):
    # 1. 데이터 시각화
    feature_columns = [
        "ActF", "SpindleSpeed", "ModalT_x", "servoload_x", "servoload_z",
        "servocurrent_x", "servocurrent_z", "SpindleLoad"
    ]
    # 피처별 시각화
    fig, axes = plt.subplots(len(feature_columns), 1, figsize=(10, len(feature_columns) * 2), sharex=True)
    for i, column in enumerate(feature_columns):
        axes[i].plot(serial_data.index, serial_data[column], color='b')
        axes[i].set_title(column, fontsize=16, weight='bold')
        axes[i].grid(True)
        axes[i].set_ylabel(column, fontsize=10)
    plt.xlabel('Index', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # suptitle 공간 확보
    
    return fig

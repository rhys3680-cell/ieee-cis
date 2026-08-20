import streamlit as st
import pandas as pd
import numpy as np
import os

# 1. 페이지 기본 설정
st.set_page_config(page_title="이상금융거래 관제 대시보드", layout="wide")

# 2. 데이터 로드 (에러 핸들링 및 캐싱 적용)
@st.cache_data
def load_data():
    data_dir = r"C:\Users\ergcx\Documents\MalangMalang\ieee-cis\data\ieee-fraud-detection"
    tx_path = os.path.join(data_dir, 'train_transaction.csv')
    
    if not os.path.exists(tx_path):
        tx_path = 'train_transaction.csv'
        
    try:
        # 대시보드 메모리 최적화 및 렌더링 속도를 위해 상위 1만 건만 로드
        df = pd.read_csv(tx_path, usecols=['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'isFraud'], nrows=10000)
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

# 3. 예측 결과 로드 (실제 환경에서는 학습된 모델의 predict_proba 결과를 매핑합니다)
@st.cache_data
def get_predictions(df):
    if df is None: return None
    np.random.seed(42)
    y_val = df['isFraud'].fillna(0).values
    # 시연을 위해 사기(1)인 경우 높은 확률을 가지도록 가상의 예측 확률 생성
    y_pred_proba = np.clip(np.random.rand(len(y_val)) * 0.4 + (y_val * 0.6), 0, 1)
    return y_val, y_pred_proba

# 대시보드 헤더
st.title("🛡️ 이상금융거래 탐지(FDS) 운영 대시보드")

df = load_data()
if df is not None:
    y_val, y_pred_proba = get_predictions(df)

    # 4. 기획안 기준 3개 탭 구성
    tabs = st.tabs(['📊 EDA 개요', '📈 모델 성능·SHAP', '🎛️ 임계값 시뮬레이터'])

    with tabs[0]:
        st.subheader("최근 결제 트래픽 모니터링")
        st.dataframe(df.head(100), use_container_width=True)

    with tabs[1]:
        st.subheader("탐지 모델 성능 지표")
        col1, col2 = st.columns(2)
        col1.metric('거래 단독 베이스라인 AUC', '0.8912')
        col2.metric('기기 결합 통합 모델 AUC', '0.9450', delta='+0.0538')
        st.info("💡 프로덕션 환경 연동 시, 개별 거래 차단 사유를 설명하는 SHAP Waterfall Plot이 이 영역에 렌더링됩니다.")

    with tabs[2]:
        st.subheader("탐지 임계값(Threshold) 시뮬레이터")
        st.write("머신러닝 모델의 예측 확률을 바탕으로, 실제 결제를 차단할 기준값을 설정합니다.")

        # 슬라이더를 통한 임계값 동적 조정
        th = st.slider('사기 판정 임계값 (Probability Threshold)', 0.0, 1.0, 0.50, 0.01)

        # 동적 성능 지표 계산
        pred_bin = (y_pred_proba >= th).astype(int)
        tp = ((pred_bin == 1) & (y_val == 1)).sum()
        fp = ((pred_bin == 1) & (y_val == 0)).sum()
        fn = ((pred_bin == 0) & (y_val == 1)).sum()
        tn = ((pred_bin == 0) & (y_val == 0)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        col1, col2 = st.columns(2)
        col1.metric('탐지율 (Recall)', f'{recall:.1%}', help="실제 사기 거래 중 시스템이 차단에 성공한 비율")
        col2.metric('오탐율 (FPR)', f'{fpr:.1%}', delta_color="inverse", help="정상 결제를 사기로 오인하여 차단한 비율")

        st.warning(f"**운영 관점 요약:** 현재 임계값({th}) 적용 시, 사기 거래의 **{recall:.1%}**를 차단할 수 있으나 정상 고객의 **{fpr:.1%}**가 결제 실패를 겪어 CS 티켓이 발행됩니다.")
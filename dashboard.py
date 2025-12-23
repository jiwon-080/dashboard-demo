# dashboard.py
from pykrx import stock
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st

# === 설정 ===
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

FEATURE_MAP = {
    "F1_ROA": "총자산이익률. 자산 대비 이익 창출 효율성",
    "F1_ROE": "자기자본이익률. 주주 자본 활용 효율성",
    "F1_Equity_Growth": "자기자본 증가율. 높을수록 안정적 성장",
    "F1_Retained_Earnings_Ratio": "이익잉여금 비율. 자본 중 잉여금 비중",
    "F1_Debt_Ratio": "부채비율. 직관적인 파산 위험 지표 (높을수록 위험)",
    "F1_Current_Ratio": "유동비율. 단기 채무 상환 능력 (낮을수록 위험)",
    "F1_Interest_Coverage": "이자보상배율. 1 미만이면 이자 감당 불가",
    "F2_KMV_DD": "부도거리(Distance to Default). 낮을수록 부도 가능성 증가",
    "F3_Z_Score": "Altman Z-score. 낮을수록 파산 가능성 증가",
    "F4_M_Score": "Beneish M-score. 높을수록 회계 조작(분식) 가능성 증가",
    "M_Short_Term_Rate": "단기금리. 상승 시 취약 기업 부담 가중",
    "M_Long_Term_Rate": "장기금리. 상승 시 투자 위축 우려",
    "M_Rate_Spread": "금리 스프레드. 장단기 차이, 경기 침체 신호",
    "M_Nominal_GDP_Growth": "명목 GDP 성장률. 낮을수록 매출 둔화",
    "M_Exchange_Rate": "환율. 상승 시 외화부채 기업 위험",
    "lex_sent_mean": "문서 평균 감성. 낮을수록 부정적 톤",
    "lex_sent_sum": "부정 감성 누적. 리스크 확대 의미",
    "lex_abs_mean": "감성 강도 절댓값. 높을수록 불확실성 증가",
    "lex_neg_cnt": "부정 문장 수. 직접적인 위험 신호",
    "koelectra_prob_default": "AI(KoELECTRA) 예측 부도 확률. 문서의 뉘앙스/맥락 반영",
    "koe_doc_cnt": "분석된 문서 수"
}

def load_data_and_model(ticker):
    code = ticker.strip() 
    try:
        today = pd.Timestamp.now().strftime("%Y%m%d")
        hist = stock.get_market_ohlcv((pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y%m%d"), today, code)
        current_price = hist['종가'].iloc[-1] if not hist.empty else 0
    except:
        current_price = 0

    # 벤치마크 데이터를 포함한 SHAP 데이터 생성
    shap_data, total_risk_score = generate_detailed_shap_data()
    indicators = determine_traffic_lights_by_group(shap_data)

    return {
        "ticker": code,
        "price": current_price,
        "risk_score": total_risk_score,
        "indicators": indicators,
        "shap_data": shap_data
    }

# dashboard.py

def generate_detailed_shap_data():
    """
    [수정] 산업군 평균 데이터 보강 및 8개 이상의 샘플 데이터 생성
    """
    raw_features = [
        {"name": "F1_Debt_Ratio", "category": "financial", "shap": 0.156, "score": 85, "normal_avg": 30, "industry_avg": 45, "val": "345%", "desc": "부채비율 매우 높음"},
        {"name": "F3_Z_Score", "category": "financial", "shap": 0.142, "score": 90, "normal_avg": 20, "industry_avg": 40, "val": "1.1", "desc": "파산 경고 구간"},
        {"name": "koelectra_prob_default", "category": "text", "shap": 0.120, "score": 82, "normal_avg": 15, "industry_avg": 30, "val": "82%", "desc": "AI 분석 부도 확률 높음"},
        {"name": "F1_Current_Ratio", "category": "financial", "shap": 0.098, "score": 75, "normal_avg": 25, "industry_avg": 35, "val": "72%", "desc": "유동성 부족"},
        {"name": "F2_KMV_DD", "category": "financial", "shap": 0.074, "score": 70, "normal_avg": 15, "industry_avg": 25, "val": "0.65", "desc": "부도거리 짧음(위험)"},
        {"name": "lex_sent_mean", "category": "text", "shap": 0.060, "score": 65, "normal_avg": 10, "industry_avg": 20, "val": "-0.45", "desc": "어조 매우 부정적"},
        {"name": "M_Short_Term_Rate", "category": "macro", "shap": 0.030, "score": 60, "normal_avg": 45, "industry_avg": 45, "val": "3.5%", "desc": "단기금리 상승 부담"},
        {"name": "lex_neg_cnt", "category": "text", "shap": 0.040, "score": 55, "normal_avg": 10, "industry_avg": 15, "val": "1,240건", "desc": "부정 단어 급증"},
        # 8개 초과를 위한 추가 데이터
        {"name": "F1_ROA", "category": "financial", "shap": 0.025, "score": 50, "normal_avg": 10, "industry_avg": 15, "val": "-2.1%", "desc": "수익성 둔화"},
        {"name": "M_Exchange_Rate", "category": "macro", "shap": 0.015, "score": 45, "normal_avg": 30, "industry_avg": 35, "val": "1,350원", "desc": "환율 변동 리스크"},
        {"name": "F1_Interest_Coverage", "category": "financial", "shap": -0.020, "score": 20, "normal_avg": 30, "industry_avg": 40, "val": "2.1배", "desc": "이자 상환 능력 양호"}
    ]
    
    sorted_features = sorted(raw_features, key=lambda x: x['shap'], reverse=True)
    return sorted_features, 82

def determine_traffic_lights_by_group(shap_data):
    score_fin = sum(f['shap'] for f in shap_data if f['category'] == 'financial')
    score_text = sum(f['shap'] for f in shap_data if f['category'] == 'text')
    score_macro = sum(f['shap'] for f in shap_data if f['category'] == 'macro')
    
    def get_color(score):
        if score > 0.08: return "red"
        elif score > 0.02: return "yellow"
        else: return "green"

    return {"financial": get_color(score_fin), "text": get_color(score_text), "macro": get_color(score_macro)}

def get_gemini_rag_analysis(data_summary, shap_data):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    top_factors_text = ""
    for item in shap_data[:5]:
        desc = FEATURE_MAP.get(item['name'], "설명 없음")
        top_factors_text += f"- {item['name']}: {item['val']} ({desc})\n"

    prompt = f"""
    당신은 기업 구조조정 및 부도 예측 전문가입니다. 
    금융 지표와 텍스트 분석 결과를 종합하여 통찰력 있는 보고서를 작성하세요.
    - 기업코드: {data_summary['ticker']}
    - AI 종합 부도 위험도: {data_summary['risk_score']}%
    [위험 요인]
    {top_factors_text}
    작성 가이드: 지표 간 인과관계를 설명하고 전문적인 어조를 유지하세요. 마지막엔 투자 주의 문구를 넣으세요.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "분석 생성 실패"
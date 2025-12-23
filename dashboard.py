# dashboard.py
from pykrx import stock
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st

# === 설정 ===
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

FEATURE_MAP = {
    # 수익성
    "F1_ROA": "총자산이익률. 자산 대비 이익 창출 효율성",
    "F1_ROE": "자기자본이익률. 주주 자본 활용 효율성",
    "F1_Equity_Growth": "자기자본 증가율. 높을수록 안정적 성장",
    "F1_Retained_Earnings_Ratio": "이익잉여금 비율. 자본 중 잉여금 비중",
    # 안정성
    "F1_Debt_Ratio": "부채비율. 직관적인 파산 위험 지표 (높을수록 위험)",
    "F1_Current_Ratio": "유동비율. 단기 채무 상환 능력 (낮을수록 위험)",
    "F1_Interest_Coverage": "이자보상배율. 1 미만이면 이자 감당 불가",
    # 시장/파산
    "F2_KMV_DD": "부도거리(Distance to Default). 낮을수록 부도 가능성 증가",
    "F3_Z_Score": "Altman Z-score. 낮을수록 파산 가능성 증가",
    "F4_M_Score": "Beneish M-score. 높을수록 회계 조작(분식) 가능성 증가",
    # 거시경제
    "M_Short_Term_Rate": "단기금리. 상승 시 취약 기업 부담 가중",
    "M_Long_Term_Rate": "장기금리. 상승 시 투자 위축 우려",
    "M_Rate_Spread": "금리 스프레드. 장단기 차이, 경기 침체 신호",
    "M_Nominal_GDP_Growth": "명목 GDP 성장률. 낮을수록 매출 둔화",
    "M_Exchange_Rate": "환율. 상승 시 외화부채 기업 위험",
    # 텍스트/AI
    "lex_sent_mean": "문서 평균 감성. 낮을수록 부정적 톤",
    "lex_sent_sum": "부정 감성 누적. 리스크 확대 의미",
    "lex_abs_mean": "감성 강도 절댓값. 높을수록 불확실성 증가",
    "lex_neg_cnt": "부정 문장 수. 직접적인 위험 신호",
    "koelectra_prob_default": "AI(KoELECTRA) 예측 부도 확률. 문서의 뉘앙스/맥락 반영",
    "koe_doc_cnt": "분석된 문서 수"
}

def load_data_and_model(ticker):
    """
    데이터 로드 및 8개 피처 그룹 기반 위험도 분석
    """
    code = ticker.strip() 
    
    # 1. 주가 데이터 (pykrx)
    try:
        today = pd.Timestamp.now().strftime("%Y%m%d")
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y%m%d")
        hist = stock.get_market_ohlcv(start_date, today, code)
        
        if not hist.empty:
            current_price = hist['종가'].iloc[-1]
            hist = hist.rename(columns={'종가': 'Close'})
        else:
            current_price = 0
            hist = pd.DataFrame()
    except:
        current_price = 0
        hist = pd.DataFrame()

    # 2. 8개 그룹 피처 기반 SHAP 데이터 생성
    shap_data, total_risk_score = generate_detailed_shap_data()

    # 3. 그룹별 신호등 판정 (SHAP 합산 기반)
    indicators = determine_traffic_lights_by_group(shap_data)

    return {
        "ticker": code,
        "price": current_price,
        "history": hist,
        "risk_score": total_risk_score,
        "indicators": indicators,
        "shap_data": shap_data
    }

def generate_detailed_shap_data():
    """
    [핵심] 사용자가 정의한 8개 그룹 피처에 대한 Mock SHAP 값 생성
    """
    # 데이터 정의: (변수명, 카테고리, SHAP기여도, 실제값, 설명)
    # SHAP > 0: 부도 위험 증가(나쁨), SHAP < 0: 안전(좋음)
    
    raw_features = [
        # --- 1. 수익성, 성장성 (Financial) ---
        {"name": "F1_ROA", "cat": "financial", "shap": 0.045, "val": "-2.5%", "desc": "자산수익률 악화"},
        {"name": "F1_ROE", "cat": "financial", "shap": 0.032, "val": "-15%", "desc": "자기자본잠식 우려"},
        {"name": "F1_Equity_Growth", "cat": "financial", "shap": 0.010, "val": "-5.2%", "desc": "자본 감소"},
        
        # --- 2. 재무 안정성 (Financial) ---
        {"name": "F1_Debt_Ratio", "cat": "financial", "shap": 0.156, "val": "345%", "desc": "부채비율 매우 높음"},
        {"name": "F1_Current_Ratio", "cat": "financial", "shap": 0.098, "val": "72%", "desc": "유동성 부족"},
        
        # --- 3. 이자 상환 능력 (Financial) ---
        {"name": "F1_Interest_Coverage", "cat": "financial", "shap": -0.020, "val": "2.1배", "desc": "이자 감당 가능"},
        
        # --- 4. 시장 기반 위험 (Financial) ---
        {"name": "F2_KMV_DD", "cat": "financial", "shap": 0.074, "val": "0.65", "desc": "부도거리 짧음(위험)"},
        
        # --- 5. 복합 파산 예측 (Financial) ---
        {"name": "F3_Z_Score", "cat": "financial", "shap": 0.142, "val": "1.1", "desc": "파산 경고 구간"},
        {"name": "F4_M_Score", "cat": "financial", "shap": 0.009, "val": "-1.8", "desc": "회계부정 가능성 낮음"},
        
        # --- 6. 거시경제 (Macro) ---
        {"name": "M_Short_Term_Rate", "cat": "macro", "shap": 0.030, "val": "3.5%", "desc": "단기금리 상승 부담"},
        {"name": "M_Exchange_Rate", "cat": "macro", "shap": 0.025, "val": "1350원", "desc": "환율 상승(비용증가)"},
        {"name": "M_Real_GDP_Growth", "cat": "macro", "shap": 0.015, "val": "1.4%", "desc": "저성장 국면"},
        
        # --- 7. 텍스트 기반 (Text) ---
        {"name": "lex_sent_mean", "cat": "text", "shap": 0.060, "val": "-0.45", "desc": "어조 매우 부정적"},
        {"name": "lex_neg_cnt", "cat": "text", "shap": 0.040, "val": "1,240건", "desc": "부정 단어 급증"},
        
        # --- 8. 딥러닝 기반 (Text) ---
        {"name": "koelectra_prob_default", "cat": "text", "shap": 0.120, "val": "82%", "desc": "AI 부도 확률 높음"},
        {"name": "koe_doc_cnt", "cat": "text", "shap": 0.005, "val": "45건", "desc": "분석 문서 수"}
    ]
    
    # 랜덤 노이즈 추가 및 리스트 변환
    shap_list = []
    total_shap = 0
    
    for item in raw_features:
        # 시뮬레이션을 위해 값 약간 변동
        noise = np.random.uniform(-0.005, 0.005)
        final_shap = item['shap'] + noise
        total_shap += final_shap
        
        shap_list.append({
            "name": item['name'],
            "category": item['cat'],
            "shap": final_shap,
            "actual": item['val'],
            "desc": item['desc']
        })

    # 총 위험 점수 (0~100점 스케일링)
    risk_score = int(np.clip((0.2 + total_shap) * 120, 10, 95))
    
    # 기여도가 큰 순서대로 정렬
    sorted_features = sorted(shap_list, key=lambda x: abs(x['shap']), reverse=True)
    
    return sorted_features, risk_score

def determine_traffic_lights_by_group(shap_data):
    """
    그룹별 SHAP 합계(Impact)를 계산하여 신호등 색상 결정
    """
    # 1. Financial (그룹 1,2,3,4,5 합산)
    score_fin = sum(f['shap'] for f in shap_data if f['category'] == 'financial')
    
    # 2. Text (그룹 7,8 합산)
    score_text = sum(f['shap'] for f in shap_data if f['category'] == 'text')
    
    # 3. Macro (그룹 6 합산)
    score_macro = sum(f['shap'] for f in shap_data if f['category'] == 'macro')
    
    def get_color(score):
        if score > 0.08: return "red"     # 위험 기여도 높음
        elif score > 0.02: return "yellow" # 주의
        else: return "green"              # 안전/양호 (음수거나 매우 낮음)

    return {
        "financial": get_color(score_fin),
        "text": get_color(score_text),
        "macro": get_color(score_macro)
    }

def get_gemini_rag_analysis(data_summary, shap_data):
    """
    [업그레이드] 컬럼 설명을 포함하여 더 똑똑한 분석 생성
    """
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    # [핵심 변경] 상위 5개 변수에 대해 '값' + '설명'을 같이 묶어서 텍스트화
    top_factors_text = ""
    for item in shap_data[:5]:
        name = item['name']
        val = item['actual']
        # 사전에 설명이 있으면 가져오고, 없으면 공란
        desc = FEATURE_MAP.get(name, "설명 없음")
        
        top_factors_text += f"- {name}: {val} (참고: {desc})\n"

    # 프롬프트 강화
    prompt = f"""
    당신은 기업 구조조정 및 부도 예측 전문가입니다. 
    금융 지표와 텍스트(공시서류 내 텍스트) 분석 결과를 종합하여 통찰력 있는 보고서를 작성하세요.

    [대상 기업 정보]
    - 기업코드: {data_summary['ticker']}
    - AI 종합 부도 위험도: {data_summary['risk_score']}%
    
    [가장 치명적인 위험 요인 Top 5 (순서대로 중요함)]
    {top_factors_text}
    
    [작성 가이드]
    1. 단순히 지표를 나열하지 말고, 지표 간의 인과관계를 설명하세요.
    (예: "단기금리(M_Short_Term_Rate)가 상승하는 가운데 이자보상배율이 낮아져 금융 비용 부담이 심각합니다.")
    2. 텍스트 지표(lex_*, koelectra_*)가 있다면, "정량적 수치뿐만 아니라 경영진의 언어적 뉘앙스에서도 불안감이 감지됨"처럼 해석하세요.
    3. 마지막에는 반드시 "본 보고서는 투자 결정에 참고 자료로만 사용되어야 하며, 최종 투자 결정은 투자자 본인의 책임하에 신중히 내려야 합니다."
    라는 경고 문구를 작성하세요.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "분석 생성 실패"
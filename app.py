# app.py
import streamlit as st
import dashboard as db

st.set_page_config(layout="wide", page_title="Advanced Risk Dashboard")

# CSS 스타일 (그대로 유지)
st.markdown("""
<style>
    .shap-row {
        display: flex; align-items: center; margin-bottom: 6px; padding: 5px;
        background-color: #ffffff; border-radius: 4px; font-size: 14px; border-bottom: 1px solid #eee;
    }
    .feature-name { flex: 2; font-weight: 600; color: #333; }
    .bar-container { flex: 3; display: flex; align-items: center; }
    .shap-bar { height: 10px; border-radius: 5px; }
    .shap-value { width: 50px; text-align: right; margin-left: 8px; font-size: 12px; color: #666; font-family: monospace;}
    .actual-val { flex: 2; text-align: right; font-size: 13px; font-weight: bold; color: #444; }
    .desc-text { flex: 2; text-align: right; color: #888; font-size: 12px; margin-left: 10px; }
    
    .metric-box {
        text-align: center; padding: 15px; border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 사이드바
st.sidebar.title("🛠️ 모델 설정")
ticker = st.sidebar.text_input("종목 코드", value="005930")
if st.sidebar.button("AI 진단 시작"):
    st.session_state['run'] = True

if st.session_state.get('run'):
    with st.spinner("딥러닝 모델(KoELECTRA) 및 재무 데이터 분석 중..."):
        data = db.load_data_and_model(ticker)
    
    # 1. 상단 헤더
    st.title(f"📊 {ticker} 통합 부도 리스크 분석")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("현재 주가", f"{data['price']:,.0f}원")
    with col2:
        risk = data['risk_score']
        st.subheader(f"🚨 부도 위험 스코어: {risk}%")
        st.progress(risk/100)
    
    st.divider()

    # 2. 신호등 (8개 그룹을 3개 카테고리로 통합)
    st.subheader("🚦 리스크 팩터 상태판")
    c1, c2, c3 = st.columns(3)
    
    ind = data['indicators']
    
    def draw_light(col, name, status, icon_char):
        colors = {"red": "#FFEBEE", "yellow": "#FFFDE7", "green": "#E8F5E9"}
        emoji = {"red": "🔴 위험", "yellow": "🟡 주의", "green": "🟢 양호"}
        
        with col:
            st.markdown(f"""
            <div class='metric-box' style='background-color: {colors[status]};'>
                <div style='font-size:30px; margin-bottom:5px;'>{icon_char}</div>
                <div style='font-weight:bold; font-size:16px;'>{name}</div>
                <div style='margin-top:5px; color:#333;'>{emoji[status]}</div>
            </div>
            """, unsafe_allow_html=True)
            
    draw_light(c1, "재무/시장 복합", ind['financial'], "💰") # 그룹 1~5
    draw_light(c2, "AI 텍스트 분석", ind['text'], "📝")      # 그룹 7~8
    draw_light(c3, "거시경제 환경", ind['macro'], "🌍")      # 그룹 6

    # 3. 상세 요인 분석 (사용자 요청 피처 반영)
    st.divider()
    st.subheader("📉 위험 기여도 상세 분석 (Top Factors)")
    
    for item in data['shap_data']:
        # 시각화 로직
        s_val = item['shap']
        width = min(abs(s_val) * 500, 100) # 길이 조절용
        color = "#ff5252" if s_val > 0 else "#448aff" # 빨강(위험) / 파랑(안전)
        
        st.markdown(f"""
        <div class='shap-row'>
            <div class='feature-name'>{item['name']}</div>
            <div class='bar-container'>
                <div class='shap-bar' style='width:{width}%; background-color:{color};'></div>
                <div class='shap-value'>{s_val:+.3f}</div>
            </div>
            <div class='actual-val'>{item['actual']}</div>
            <div class='desc-text'>{item['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 4. Gemini 리포트
    st.divider()
    st.subheader("✨ Generative AI 리포트")
    report = db.get_gemini_rag_analysis(data, data['shap_data'])
    st.info(report)
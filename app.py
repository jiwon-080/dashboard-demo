# app.py
import streamlit as st
st.set_page_config(layout="wide", page_title="Advanced Risk Dashboard")

import dashboard as db
import plotly.graph_objects as go
import pandas as pd

# CSS 스타일 (그대로 유지)
st.markdown("""
<style>
    .shap-row { display: flex; align-items: center; margin-bottom: 6px; padding: 5px; background-color: #ffffff; border-radius: 4px; font-size: 14px; border-bottom: 1px solid #eee; }
    .feature-name { flex: 2; font-weight: 600; color: #333; }
    .bar-container { flex: 3; display: flex; align-items: center; }
    .shap-bar { height: 10px; border-radius: 5px; }
    .shap-value { width: 50px; text-align: right; margin-left: 8px; font-size: 12px; color: #666; font-family: monospace;}
    .actual-val { flex: 2; text-align: right; font-size: 13px; font-weight: bold; color: #444; }
    .desc-text { flex: 2; text-align: right; color: #888; font-size: 12px; margin-left: 10px; }
    .metric-box { text-align: center; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# 사이드바
st.sidebar.title("🛠️ 모델 설정")
ticker = st.sidebar.text_input("종목 코드")
if st.sidebar.button("AI 진단 시작"):
    st.session_state['run'] = True

if st.session_state.get('run'):
    with st.spinner("데이터 분석 중..."):
        data = db.load_data_and_model(ticker)
        if data is None:
            st.error("⚠️ 해당 종목 코드를 찾을 수 없습니다. 다시 확인해주세요.")
            st.stop()

    st.title(f"📊 {ticker} 통합 부도 리스크 분석")
    col_h1, col_h2 = st.columns([1, 2])
    with col_h1: st.metric("현재 주가", f"{data['price']:,.0f}원")
    with col_h2:
        risk = data['risk_score']
        st.subheader(f"🚨 부도 위험 스코어: {risk}%")
        st.progress(risk/100)
    
    st.divider()
    st.subheader("🚦 리스크 팩터 상태판")
    c1, c2, c3 = st.columns(3)
    ind = data['indicators']
    def draw_light(col, name, status, icon_char):
        colors = {"red": "#FFEBEE", "yellow": "#FFFDE7", "green": "#E8F5E9"}
        emoji = {"red": "🔴 위험", "yellow": "🟡 주의", "green": "🟢 양호"}
        with col: st.markdown(f"<div class='metric-box' style='background-color: {colors[status]};'><h3>{icon_char}</h3><b>{name}</b><p>{emoji[status]}</p></div>", unsafe_allow_html=True)
    draw_light(c1, "재무/시장 복합", ind['financial'], "💰")
    draw_light(c2, "AI 텍스트 분석", ind['text'], "📝")
    draw_light(c3, "거시경제 환경", ind['macro'], "🌍")

    
    # 3. 다차원 리스크 분석 (상위 8개 피처 수준 비교)
    st.divider()
    st.subheader("📊 벤치마크 리스크 프로파일 (정상/산업 평균 대비 피처 수준)")
    
    df_all = pd.DataFrame(data['shap_data'])
    plot_df = df_all.head(8) # 다중지능 그래프는 상위 8개 유지 (가독성)
    
    categories = plot_df['name'].tolist()
    company_scores = plot_df['score'].tolist()
    normal_scores = plot_df['normal_avg'].tolist()
    industry_scores = plot_df['industry_avg'].tolist() # 산업군 평균 리스트 추가
    
    hover_labels = [f"원본값: {row['val']}<br>설명: {db.FEATURE_MAP.get(row['name'], '')}" for _, row in plot_df.iterrows()]

    col_bar, col_radar = st.columns(2)

    with col_bar:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=categories, y=company_scores, name='대상 기업', marker_color='red', customdata=hover_labels, hovertemplate="<b>%{x}</b><br>위험 점수: %{y}점<br>%{customdata}<extra></extra>"))
        fig_bar.add_trace(go.Bar(x=categories, y=normal_scores, name='정상 평균', marker_color='green', opacity=0.5))
        fig_bar.add_trace(go.Bar(x=categories, y=industry_scores, name='산업 평균', marker_color='orange', opacity=0.5)) # 바 차트에 산업 평균 추가
        fig_bar.update_layout(title="주요 위험 요인 수준 비교", barmode='group', height=500)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_radar:
        def wrap(l): return l + [l[0]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=wrap(normal_scores), theta=wrap(categories), fill='toself', name='정상 평균', line_color='green', opacity=0.3))
        fig_radar.add_trace(go.Scatterpolar(r=wrap(industry_scores), theta=wrap(categories), name='산업 평균', line=dict(color='orange', dash='dash'))) # 레이더에 산업 평균 추가
        fig_radar.add_trace(go.Scatterpolar(r=wrap(company_scores), theta=wrap(categories), name='분석 대상', line=dict(color='red', width=4), customdata=wrap(hover_labels), hovertemplate="<b>%{theta}</b><br>위험 점수: %{r}점<br>%{customdata}<extra></extra>"))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="다차원 리스크 균형도", height=500)
        st.plotly_chart(fig_radar, use_container_width=True)

    # 4. SHAP 전체 출력 (기여도 그래프)
    st.divider()
    st.subheader("📉 전체 요인별 부도 기여도 (SHAP)")
    st.caption("※ 모든 분석 피처의 기여도를 출력합니다.")
    
    # df_all 전체를 사용하여 8개 이상의 지표 출력
    fig_shap_all = go.Figure(go.Bar(
        y=df_all['name'], x=df_all['shap'], orientation='h',
        marker_color=['#ff5252' if x > 0 else '#448aff' for x in df_all['shap']],
        customdata=[db.FEATURE_MAP.get(n, "") for n in df_all['name']],
        hovertemplate="<b>%{y}</b><br>SHAP 기여도: %{x:+.3f}<br>%{customdata}<extra></extra>"
    ))
    # 데이터 양에 따라 그래프 높이가 자동 조절되도록 설정 가능 (예: len(df_all) * 30)
    fig_shap_all.update_layout(height=max(400, len(df_all) * 35), yaxis={'categoryorder':'total ascending'}, xaxis_title="SHAP 기여도")
    st.plotly_chart(fig_shap_all, use_container_width=True)

    # (상세 리스트 및 Gemini 리포트 생략)
    st.divider()
    st.subheader("✨ Generative AI 리포트")
    st.info(db.get_gemini_rag_analysis(data, data['shap_data']))
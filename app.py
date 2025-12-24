import streamlit as st
st.set_page_config(layout="wide", page_title="잡았다 요놈! Risk Dashboard")

import dashboard as db
import plotly.graph_objects as go
import pandas as pd

# CSS 스타일 (사용자님 원본 유지)
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
st.sidebar.title("🔎 기업 검색")
ticker_input = st.sidebar.text_input("종목 코드", value="005930") # 입력값 유지 위해 value 추가
if st.sidebar.button("진단 시작"):
    st.session_state['run'] = True
    st.session_state['current_ticker'] = ticker_input

if st.session_state.get('run'):
    with st.spinner("데이터 분석 중..."):
        # ticker_input 대신 세션의 ticker 사용 (새로고침 방지)
        target_ticker = st.session_state.get('current_ticker', ticker_input)
        data = db.load_data_and_model(target_ticker)
        
        if data is None:
            st.error("⚠️ 해당 종목 코드를 찾을 수 없습니다. 다시 확인해주세요.")
            st.stop()

    # =========================================================================
    # [수정] 변수 정의를 맨 위로 올려서 에러 방지 (UI는 건드리지 않음)
    # =========================================================================
    shap_data = data['shap_data']     # 이제 shap_data 사용 가능
    df_all = pd.DataFrame(shap_data)  # 이제 df_all 사용 가능
    risk = data['risk_score']         # risk 변수 정의
    
    # -------------------------------------------------------------------------
    # [UI 복구] 사용자님 원래 디자인 (헤더, 프로그레스바, 신호등)
    # -------------------------------------------------------------------------
    st.title(f"📊 {data['ticker']} 통합 부도 리스크 분석") # ticker 변수 대신 data['ticker'] 사용
    
    col_h1, col_h2 = st.columns([1, 2])
    with col_h1: 
        st.metric("현재 주가", f"{data['price']:,.0f}원")
    with col_h2:
        st.subheader(f"🚨 부도 위험 스코어: {risk}%") # 복구 완료
        st.progress(risk/100) # 복구 완료
    
    st.divider()
    
    # 신호등 섹션 복구
    st.subheader("🚦 리스크 팩터 상태판")
    c1, c2, c3 = st.columns(3)
    ind = data['indicators']
    
    def draw_light(col, name, status, icon_char):
        colors = {"red": "#FFEBEE", "yellow": "#FFFDE7", "green": "#E8F5E9"}
        emoji = {"red": "🔴 위험", "yellow": "🟡 주의", "green": "🟢 양호"}
        with col: 
            st.markdown(f"<div class='metric-box' style='background-color: {colors[status]};'><h3>{icon_char}</h3><b>{name}</b><p>{emoji[status]}</p></div>", unsafe_allow_html=True)
            
    draw_light(c1, "재무/시장 복합", ind['financial'], "💰")
    draw_light(c2, "AI 텍스트 분석", ind['text'], "📝")
    draw_light(c3, "거시경제 환경", ind['macro'], "🌍")

    # --------------------------------------------------------------------------------
    # [3. 7대 핵심 건전성 분석] (여기는 아까 요청하신 대로 교체된 버전 유지)
    # --------------------------------------------------------------------------------
    st.divider()
    st.subheader("📊 7대 핵심 건전성 분석")
    st.caption("※ 49개 세부 지표를 7가지 핵심 역량으로 그룹화하여 분석한 결과입니다. (점수가 높을수록 우량/안전)")

    # 1. 매핑 로직
    def get_category(name):
        name = name.lower()
        if any(x in name for x in ['roa', 'roe', 'interest_coverage']): return '💰 수익성'
        if any(x in name for x in ['debt', 'current_ratio', 'retained']): return '🛡️ 재무안정성'
        if any(x in name for x in ['equity_growth']): return '📈 성장성'
        if any(x in name for x in ['kmv', 'z_score', 'm_score']): return '🔎 탐지모델'
        if name.startswith('m_'): return '🌍 거시환경'
        if 'prob' in name: return '📝 NLP분석'
        if 'lex' in name: return '❤️ 감성분석'
        return '기타'

    # 2. 데이터 그룹화
    radar_data = {} 
    target_categories = ['💰 수익성', '🛡️ 재무안정성', '📈 성장성', '🔎 탐지모델', '🌍 거시환경', '📝 NLP분석', '❤️ 감성분석']
    
    for cat in target_categories:
        radar_data[cat] = {'company': [], 'industry': [], 'normal': []}

    for item in shap_data:
        cat = get_category(item['name'])
        if cat in radar_data:
            radar_data[cat]['company'].append(item['score'])
            radar_data[cat]['industry'].append(item['industry_avg'])
            radar_data[cat]['normal'].append(item['normal_avg'])

    # 3. 평균 계산
    final_cats = []
    c_scores, i_scores, n_scores = [], [], []

    for cat in target_categories:
        final_cats.append(cat)
        vals_c = radar_data[cat]['company']
        c_scores.append(sum(vals_c)/len(vals_c) if vals_c else 50)
        vals_i = radar_data[cat]['industry']
        i_scores.append(sum(vals_i)/len(vals_i) if vals_i else 50)
        vals_n = radar_data[cat]['normal']
        n_scores.append(sum(vals_n)/len(vals_n) if vals_n else 50)

    # 4. 차트 그리기
    col_bar, col_radar = st.columns(2)

    with col_bar:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=final_cats, y=c_scores, 
            name='대상 기업', marker_color='#2962ff',
            text=[f"{s:.0f}" for s in c_scores], textposition='auto',
            hovertemplate="<b>%{x}</b><br>건전성: %{y:.1f}점<extra></extra>"
        ))
        fig_bar.add_trace(go.Bar(x=final_cats, y=n_scores, name='정상 평균', marker_color='green', opacity=0.5))
        fig_bar.add_trace(go.Bar(x=final_cats, y=i_scores, name='산업 평균', marker_color='orange', opacity=0.5))
        
        fig_bar.update_layout(
            title="분야별 건전성 점수 비교", barmode='group',
            yaxis=dict(title="점수 (100점 만점)", range=[0, 100]),
            height=400, legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_radar:
        def wrap(l): return l + [l[0]] 
        
        fig_radar = go.Figure()
        
        # 1. 정상/산업 (배경)
        # [수정 3] showlegend=True로 변경 (기본값이 True이므로 False 옵션 삭제)
        fig_radar.add_trace(go.Scatterpolar(
            r=wrap(n_scores), theta=wrap(final_cats), 
            name='정상 평균', 
            line=dict(color='green', dash='solid'),       # 진한 녹색 선 (두께 2)
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=wrap(i_scores), theta=wrap(final_cats), 
            name='산업 평균', 
            line=dict(color='orange', dash='dash')
        ))
        
        # 2. 내 기업 (메인)
        fig_radar.add_trace(go.Scatterpolar(
            r=wrap(c_scores), theta=wrap(final_cats), 
            name='분석 대상', 
            fill='toself', 
            line=dict(color='#2962ff', width=3), 
            opacity=0.4,
            hovertemplate="<b>%{theta}</b><br>건전성: %{r:.1f}점<extra></extra>"
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], ticksuffix="점", gridcolor='#eee'),
                angularaxis=dict(gridcolor='#eee'),
                bgcolor='white'
            ),
            title="다차원 건전성 균형도",
            height=400,
            margin=dict(t=40, b=40, l=40, r=40),
            legend=dict(orientation="h", y=-0.15) # 범례 위치 조정
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 4. SHAP 전체 출력 (토글 적용 + 높이 자동 조절)
    st.divider()
    st.subheader("📉 전체 요인별 상세 분석")
    st.caption("※ 클릭하면 모든 49개 지표의 기여도를 볼 수 있습니다.")

    # [수정] st.expander를 사용하여 내용을 숨김/펼침 처리
    with st.expander("🔍 전체 지표 기여도 보기 (Click to Open)", expanded=False):
        
        # [핵심] 데이터 개수(len(df_all))에 따라 높이를 자동으로 계산 (행당 30픽셀)
        # 이렇게 하면 지표가 아무리 많아도 스크롤이 생기거나 잘리지 않고 길게 나옵니다.
        dynamic_height = max(500, len(df_all) * 30)
        
        fig_shap_all = go.Figure(go.Bar(
            y=df_all['name'], 
            x=df_all['shap'], 
            orientation='h',
            marker_color=['#ff5252' if x > 0 else '#2962ff' for x in df_all['shap']], # 위험:빨강, 안전:파랑
            customdata=[db.FEATURE_MAP.get(n, n) for n in df_all['name']],
            hovertemplate="<b>%{customdata}</b> (%{y})<br>기여도: %{x:+.4f}<extra></extra>"
        ))
        
        fig_shap_all.update_layout(
            height=dynamic_height,  # 높이 자동 적용
            yaxis=dict(
                dtick=1, # 모든 항목 라벨 표시
                categoryorder='total ascending', # 값 크기순 정렬
                automargin=True # 라벨 길어도 잘리지 않게 여백 자동
            ),
            xaxis_title="부도 위험 기여도 (SHAP Value)",
            margin=dict(l=10, r=10, t=30, b=50)
        )
        st.plotly_chart(fig_shap_all, use_container_width=True)

    # 5. Gemini 리포트
    st.divider()
    st.subheader("✨ Generative AI 리포트")
    # data와 shap_data를 넘겨줍니다
    st.info(db.get_gemini_rag_analysis(data, shap_data))
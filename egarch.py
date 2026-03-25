import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
import os
import time

# --- CẤU HÌNH API KEY VNSTOCK ---
os.environ['VNSTOCK_API_KEY'] = 'vnstock_17b56a86b930db526e25e8de447a0bfd'
from vnstock import Quote

# Cấu hình trang Streamlit
st.set_page_config(page_title="VN-Index Fear & Greed Score", layout="wide", initial_sidebar_state="expanded")

# ================= HÀM TẢI DỮ LIỆU =================
@st.cache_data(ttl=86400, show_spinner=False)
def load_data(tickers, days=1095): # Tải hẳn 3 năm (1095 ngày) cho thoải mái
    CACHE_FILE = "quant_risk_cache.csv"
    
    # 1. KIỂM TRA VÀ ĐỌC TỪ FILE CACHE
    if os.path.exists(CACHE_FILE):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        # Nếu file đã được cập nhật trong ngày hôm nay -> Dùng luôn
        if file_mod_time.date() == datetime.now().date():
            st.sidebar.success(f"⚡ Đã tải dữ liệu từ Cache lúc {file_mod_time.strftime('%H:%M:%S')}")
            df_merged = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
            
            df_vnindex = df_merged[['VNINDEX']].copy()
            df_stocks = df_merged.drop(columns=['VNINDEX'])
            return df_vnindex, df_stocks

    # 2. TẢI MỚI TỪ API (CÓ RATE LIMIT & CHUẨN HÓA THỜI GIAN)
    st.sidebar.info("🔄 Đang tải dữ liệu mới từ API (khoảng 3-4 phút)...")
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    all_data = []
    
    # 2.1 Tải VN-Index
    try:
        quote_vnindex = Quote(symbol='VNINDEX', source='KBS')
        df_vnindex = quote_vnindex.history(start=start_date, end=end_date, interval='1D')
        df_vnindex = df_vnindex[['time', 'close']].rename(columns={'close': 'VNINDEX'}).set_index('time')
        
        # CHUẨN HÓA INDEX & VÁ LỖI TRÙNG LẶP (Lấy giá chốt phiên - last)
        df_vnindex.index = pd.to_datetime(df_vnindex.index).normalize().tz_localize(None)
        df_vnindex = df_vnindex[~df_vnindex.index.duplicated(keep='last')]
        all_data.append(df_vnindex)
    except Exception as e:
        st.error(f"❌ Lỗi tải VNINDEX: {e}")
        return None, None

    # 2.2 Tải 200 mã cổ phiếu
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    for i, symbol in enumerate(tickers):
        try:
            status_text.text(f"Đang tải: {symbol} ({i+1}/{len(tickers)})")
            quote = Quote(symbol=symbol, source='KBS')
            df = quote.history(start=start_date, end=end_date, interval='1D')
            
            if df is not None and not df.empty:
                df_close = df[['time', 'close']].rename(columns={'close': symbol}).set_index('time')
                
                # CHUẨN HÓA INDEX TƯƠNG TỰ VN-INDEX & LỌC TRÙNG LẶP (keep='last')
                df_close.index = pd.to_datetime(df_close.index).normalize().tz_localize(None)
                df_close = df_close[~df_close.index.duplicated(keep='last')]
                all_data.append(df_close)
                
            time.sleep(1) # Chống bị block IP
            
        except Exception as e:
            print(f"Lỗi tải {symbol}: {e}")
            time.sleep(1)
            continue
            
        progress_bar.progress((i + 1) / len(tickers))
        
    status_text.empty()
    progress_bar.empty()
    
    # 3. GỘP DỮ LIỆU VÀ LƯU CACHE
    if len(all_data) < 2: 
        st.error("❌ Không đủ dữ liệu để tính toán.")
        return None, None
        
    df_merged = pd.concat(all_data, axis=1).sort_index().ffill()
    df_merged.to_csv(CACHE_FILE)
    st.sidebar.success("✅ Đã cập nhật xong Cache ngày hôm nay.")
    
    return df_merged[['VNINDEX']].copy(), df_merged.drop(columns=['VNINDEX'])

# ================= HÀM TÍNH TOÁN CÁC CHỈ BÁO QUANT =================
@st.cache_data(show_spinner=False)
def calculate_quant_metrics(df_vnindex, df_stocks):
    # Tính lợi nhuận hàng ngày (Không dropna cho toàn bộ cổ phiếu)
    market_ret = df_vnindex['VNINDEX'].pct_change()
    stocks_ret = df_stocks.pct_change()
    
    # Đồng bộ Index theo VN-Index
    stocks_ret = stocks_ret.reindex(market_ret.index)
    
    market_ret_dropna = market_ret.dropna()
    
    # Kiểm tra chốt chặn
    if market_ret_dropna.empty or len(market_ret_dropna) < 60:
        st.error(f"❌ Lỗi: Dữ liệu VN-Index chỉ có {len(market_ret_dropna)} dòng.")
        st.stop()
        
    # Ép dữ liệu thành mảng NumPy liên tục chuẩn float64 (Vá lỗi Cython Buffer Mismatch)
    clean_y = np.ascontiguousarray(market_ret_dropna.values * 100, dtype=np.float64)
    
    # 1. EGARCH(1,1,1) Volatility
    try:
        model = arch_model(clean_y, vol='EGARCH', p=1, o=1, q=1, dist='t')
        res = model.fit(update_freq=0, disp='off')
        egarch_vol = pd.Series((res.conditional_volatility / 100) * np.sqrt(252), index=market_ret_dropna.index)
    except Exception as e:
        st.error(f"❌ Lỗi khi chạy mô hình EGARCH: {e}")
        st.stop()
        
    # 2. Rolling Skewness (60 ngày)
    rolling_skew = market_ret_dropna.rolling(window=60).skew().ffill()
    
    # 3. Upside & Downside Correlation
    market_down = market_ret_dropna.where(market_ret_dropna < 0, np.nan)
    market_up = market_ret_dropna.where(market_ret_dropna > 0, np.nan)
    
    stock_down = stocks_ret.where(market_ret_dropna < 0, np.nan)
    stock_up = stocks_ret.where(market_ret_dropna > 0, np.nan)
    
    downside_corr = stock_down.rolling(window=60, min_periods=15).corr(market_down).median(axis=1).ffill()
    upside_corr = stock_up.rolling(window=60, min_periods=15).corr(market_up).median(axis=1).ffill()
    
    # 4. Cross-Sectional Volatility (Idiosyncratic Risk)
    roll_var_m = market_ret_dropna.rolling(window=60).var()
    roll_cov = stocks_ret.rolling(window=60).cov(market_ret_dropna)
    roll_beta = roll_cov.div(roll_var_m, axis=0)
    roll_var_i = stocks_ret.rolling(window=60).var()
    
    idio_variance = roll_var_i - (roll_beta ** 2).multiply(roll_var_m, axis=0)
    idio_variance = idio_variance.clip(lower=0)
    csv_index = idio_variance.median(axis=1).ffill()
    
    metrics_df = pd.DataFrame({
        'Market_Return': market_ret_dropna,
        'EGARCH_Vol': egarch_vol,
        'Skewness': rolling_skew,
        'Downside_Corr': downside_corr,
        'Upside_Corr': upside_corr,
        'CSV_Index': csv_index
    }).dropna()
    
    return metrics_df

# ================= HÀM CHẤM ĐIỂM FEAR & GREED =================
def calculate_risk_score(metrics_df):
    df = metrics_df.copy()
    scaler = MinMaxScaler()
    
    df['Vol_Norm'] = scaler.fit_transform(df[['EGARCH_Vol']])
    df['Down_Corr_Norm'] = scaler.fit_transform(df[['Downside_Corr']])
    df['Up_Corr_Norm'] = scaler.fit_transform(df[['Upside_Corr']])
    df['Skew_Clipped'] = df['Skewness'].clip(-1, 1)
    
    scores = []
    for _, row in df.iterrows():
        vol = row['Vol_Norm']
        skew = row['Skew_Clipped']
        down_corr = row['Down_Corr_Norm']
        up_corr = row['Up_Corr_Norm']
        
        panic_pull = vol * down_corr * abs(min(skew, 0)) 
        fomo_push = vol * up_corr * max(skew, 0)         
        
        if panic_pull > fomo_push:
            score = 50 - (50 * panic_pull)  
        else:
            score = 50 + (50 * fomo_push)   
            
        scores.append(score)
        
    df['Risk_Score'] = scores
    df['Risk_Score'] = df['Risk_Score'].ewm(span=3, adjust=False).mean()
    return df

# ================= GIAO DIỆN STREAMLIT =================
st.sidebar.title("Cài đặt Hệ thống")

# Nút Xóa Cache & Ép tải lại từ API
if st.sidebar.button("🔄 Cập nhật Dữ liệu EOD", use_container_width=True):
    st.cache_data.clear()
    cache_file = "quant_risk_cache.csv"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        
    st.sidebar.success("Đã xóa dữ liệu cũ! Đang tiến hành tải lại từ API...")
    time.sleep(1)
    st.rerun()

st.title("🎯 HỆ THỐNG ĐO LƯỜNG TÂM LÝ THỊ TRƯỜNG (FEAR & GREED)")
st.markdown("*Nhận diện dòng tiền Hoảng loạn (Extreme Fear), Hưng phấn (Extreme Greed) và Môi trường Phân hóa.*")

try:
    df_tickers = pd.read_csv('danh_sach_200_ma.csv')
    tickers = df_tickers['Ticker'].tolist()
except FileNotFoundError:
    st.error("Không tìm thấy file danh_sach_200_ma.csv")
    st.stop()

with st.spinner('Đang tải dữ liệu và chạy mô hình Định lượng...'):
    df_vnindex, df_stocks = load_data(tickers)

if df_vnindex is not None and df_stocks is not None:
    metrics_df = calculate_quant_metrics(df_vnindex, df_stocks)
    scored_df = calculate_risk_score(metrics_df)
    
    latest = scored_df.iloc[-1]
    prev = scored_df.iloc[-2]
    
    current_score = latest['Risk_Score']
    score_change = current_score - prev['Risk_Score']
    
    latest_date_str = scored_df.index[-1].strftime("%d/%m/%Y")
    
    # --- PHẦN 1: ĐỒNG HỒ ĐO TÂM LÝ (FEAR & GREED INDEX) ---
    st.markdown(f"### 🚦 VN-Index Fear & Greed Score (Chốt phiên: **{latest_date_str}**)")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if current_score <= 20:
            status, color, desc = "EXTREME FEAR", "red", "Sợ hãi tột độ. Margin call diện rộng. Đám đông hoảng loạn."
        elif current_score >= 80:
            status, color, desc = "EXTREME GREED", "green", "Tham lam quá đà. Dòng tiền bất chấp rủi ro (FOMO)."
        elif 40 <= current_score <= 60:
            status, color, desc = "STOCK PICKING", "gray", "Trạng thái tĩnh lặng. Tập trung tìm Alpha ở từng cổ phiếu."
        else:
            status, color, desc = "TRANSITION", "orange", "Chuyển pha tâm lý. Động lượng dòng tiền đang thay đổi."
            
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"<b>{status}</b><br><span style='font-size:0.8em;color:gray'>{desc}</span>"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 20], 'color': "rgba(255, 0, 0, 0.2)"},
                    {'range': [20, 40], 'color': "rgba(255, 165, 0, 0.2)"},
                    {'range': [40, 60], 'color': "rgba(128, 128, 128, 0.2)"},
                    {'range': [60, 80], 'color': "rgba(173, 255, 47, 0.2)"},
                    {'range': [80, 100], 'color': "rgba(0, 128, 0, 0.2)"}],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': current_score}
            }
        ))
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col2:
        st.markdown(f"#### 🔍 Trạng thái Dòng tiền (End of Day):")
        
        # Định dạng dấu + tường minh cho Delta
        st.metric(
            label="🎯 Chỉ số Tham lam & Sợ hãi hiện tại", 
            value=f"{current_score:.1f} / 100", 
            delta=f"{score_change:+.1f} điểm so với hôm qua", 
            delta_color="normal" 
        )
        st.markdown("---")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("EGARCH Volatility", f"{latest['EGARCH_Vol']:.2%}", f"{(latest['EGARCH_Vol'] - prev['EGARCH_Vol']):+.2%}", delta_color="inverse")
        m2.metric("Rolling Skewness", f"{latest['Skewness']:.2f}", f"{(latest['Skewness'] - prev['Skewness']):+.2f}")
        m3.metric("Downside Corr", f"{latest['Downside_Corr']:.2f}", f"{(latest['Downside_Corr'] - prev['Downside_Corr']):+.2f}", delta_color="inverse")
        m4.metric("Upside Corr", f"{latest['Upside_Corr']:.2f}", f"{(latest['Upside_Corr'] - prev['Upside_Corr']):+.2f}")
        
        st.info("""
        *💡 **Cơ chế kéo/đẩy của Đồng hồ:** Nếu Biến động (Vol) vọt lên + Skewness âm + Downside Corr bám sát 1 => Hệ thống sẽ nhận diện "Áp lực bán tháo" và giật kim về 0 (Sợ hãi). 
        Ngược lại, nếu Skewness dương và Upside Corr bám sát 1 => "Áp lực đua lệnh" đẩy kim lên 100 (Tham lam).*
        """)

    # --- PHẦN 2: TRỰC QUAN HÓA BỘ CHỈ BÁO THÀNH PHẦN ---
    st.markdown("### 📈 Biểu đồ Phân tích Định hướng Dòng tiền")
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        specs=[[{"secondary_y": True}], 
               [{"secondary_y": True}], 
               [{"secondary_y": False}]],
        subplot_titles=('1. VN-Index vs EGARCH Volatility (Systemic Risk)', 
                        '2. Định hướng rủi ro (Skewness & Correlation)', 
                        '3. Cross-Sectional Volatility (%) - Cơ hội Stock Picking')
    )

    colors = ['seagreen' if val > 0 else 'crimson' for val in scored_df['Market_Return']]
    fig.add_trace(go.Bar(x=scored_df.index, y=scored_df['Market_Return'], name='VNINDEX Return', marker_color=colors, opacity=0.6), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=scored_df.index, y=scored_df['EGARCH_Vol'], name='EGARCH Volatility', line=dict(color='purple', width=2)), row=1, col=1, secondary_y=True)
    
    fig.add_trace(go.Scatter(x=scored_df.index, y=scored_df['Downside_Corr'], name='Downside Corr', line=dict(color='red', width=1.5)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=scored_df.index, y=scored_df['Upside_Corr'], name='Upside Corr', line=dict(color='green', width=1.5)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=scored_df.index, y=scored_df['Skewness'], name='Rolling Skewness', line=dict(color='black', dash='dot')), row=2, col=1, secondary_y=True)

    csv_vol_percent = np.sqrt(scored_df['CSV_Index']) * 100 
    fig.add_trace(go.Scatter(x=scored_df.index, y=csv_vol_percent, name='CSV (%)', line=dict(color='royalblue', width=2), fill='tozeroy'), row=3, col=1)

    fig.update_layout(
        height=850,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Daily Return", tickformat=".1%", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="EGARCH Vol", tickformat=".1%", row=1, col=1, secondary_y=True)
    
    fig.update_yaxes(title_text="Correlation (-1 to 1)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Skewness", row=2, col=1, secondary_y=True)
    
    fig.update_yaxes(title_text="CSV Volatility (%)", tickformat=".1f", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

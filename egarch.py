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
# Thay thế bằng API Key của bạn (nếu có)
os.environ['VNSTOCK_API_KEY'] = 'vnstock_17b56a86b930db526e25e8de447a0bfd'
from vnstock import Quote

# Cấu hình trang Streamlit
st.set_page_config(page_title="VN-Index Fear & Greed Score", layout="wide", initial_sidebar_state="expanded")

# ================= HÀM TẢI DỮ LIỆU =================
@st.cache_data(ttl=86400, show_spinner=False)
def load_data(tickers, days=1095): # Tải hẳn 3 năm (1095 ngày) cho thoải mái
    CACHE_FILE = "quant_risk_cache.csv"
    
    # ==========================================
    # BƯỚC 1: KIỂM TRA VÀ ĐỌC TỪ FILE CACHE
    # ==========================================
    if os.path.exists(CACHE_FILE):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        # Nếu file đã được cập nhật trong ngày hôm nay -> Dùng luôn
        if file_mod_time.date() == datetime.now().date():
            st.sidebar.success(f"⚡ Đã tải dữ liệu từ Cache lúc {file_mod_time.strftime('%H:%M:%S')}")
            df_merged = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
            
            df_vnindex = df_merged[['VNINDEX']].copy()
            df_stocks = df_merged.drop(columns=['VNINDEX'])
            return df_vnindex, df_stocks

    # ==========================================
    # BƯỚC 2: TẢI MỚI TỪ API (CÓ RATE LIMIT & CHUẨN HÓA THỜI GIAN)
    # ==========================================
    st.sidebar.info("🔄 Đang tải dữ liệu mới từ API (khoảng 3-4 phút)...")
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    all_data = []
    
    # 2.1 Tải VN-Index
   try:
        quote_vnindex = Quote(symbol='VNINDEX', source='KBS')
        df_vnindex = quote_vnindex.history(start=start_date, end=end_date, interval='1D')
        df_vnindex = df_vnindex[['time', 'close']].rename(columns={'close': 'VNINDEX'}).set_index('time')
        
        # CHUẨN HÓA INDEX
        df_vnindex.index = pd.to_datetime(df_vnindex.index).normalize().tz_localize(None)
        
        # ---> DÒNG CẦN THÊM VÀO: Vá lỗi Duplicate Index của VN-Index <---
        df_vnindex = df_vnindex[~df_vnindex.index.duplicated(keep='first')]
        
        all_data.append(df_vnindex)
        print("   -> VNINDEX: OK")
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
                
                # CHUẨN HÓA INDEX TƯƠNG TỰ VN-INDEX
                df_close.index = pd.to_datetime(df_close.index).normalize().tz_localize(None)
                
                # Loại bỏ các ngày bị trùng lặp (nếu có do API lỗi)
                df_close = df_close[~df_close.index.duplicated(keep='first')]
                
                all_data.append(df_close)
                
            time.sleep(1) # Chống bị block IP
            
        except Exception as e:
            print(f"Lỗi tải {symbol}: {e}")
            time.sleep(1)
            continue
            
        progress_bar.progress((i + 1) / len(tickers))
        
    status_text.empty()
    progress_bar.empty()
    
    # ==========================================
    # BƯỚC 3: GỘP DỮ LIỆU VÀ LƯU CACHE
    # ==========================================
    if len(all_data) < 2: # Ít nhất phải có VN-Index và 1 vài mã
        st.error("❌ Không đủ dữ liệu để tính toán.")
        return None, None
        
    # Nối tất cả lại bằng Outer Join để giữ nguyên cấu trúc
    df_merged = pd.concat(all_data, axis=1)
    df_merged.sort_index(inplace=True)
    
    # Forward fill (ffill) để lấp các ngày mã cổ phiếu bị mất thanh khoản / ngừng giao dịch
    df_merged = df_merged.ffill()
    
    # Lưu ra file cứng
    df_merged.to_csv(CACHE_FILE)
    st.sidebar.success("✅ Đã cập nhật xong Cache ngày hôm nay.")
    
    df_vnindex_final = df_merged[['VNINDEX']].copy()
    df_stocks_final = df_merged.drop(columns=['VNINDEX'])
    
    return df_vnindex_final, df_stocks_final

# ================= HÀM TÍNH TOÁN CÁC CHỈ BÁO QUANT =================
@st.cache_data(show_spinner=False)
def calculate_quant_metrics(df_vnindex, df_stocks):
    # 1. Kiểm tra xem data gốc có bị rỗng không
    if df_vnindex is None or df_vnindex.empty:
        st.error("❌ Lỗi: Dữ liệu VN-Index bị rỗng. Không thể chạy mô hình.")
        st.stop()
        
    if df_stocks is None or df_stocks.empty:
        st.error("❌ Lỗi: Dữ liệu 200 mã cổ phiếu bị rỗng. Vui lòng kiểm tra lại API.")
        st.stop()

    # 2. Tính lợi nhuận hàng ngày
    market_ret = df_vnindex['VNINDEX'].pct_change().dropna()
    stocks_ret = df_stocks.pct_change()
    stocks_ret = stocks_ret.reindex(market_ret.index)

    # 3. Đồng bộ dữ liệu (Inner Join)
    market_ret, stocks_ret = market_ret.align(stocks_ret, join='inner', axis=0)
    
    # --- CHỐT CHẶN QUAN TRỌNG NHẤT ---
    # Kiểm tra xem sau khi join xong, data có bị bốc hơi không
    if market_ret.empty or len(market_ret) < 60:
        st.error(f"❌ Lỗi: Dữ liệu VN-Index chỉ có {len(market_ret)} dòng.")
        st.stop()
    # ---------------------------------
    
    # Nhân 100 để thư viện arch dễ hội tụ hơn (tránh lỗi scale)
    market_ret_scaled = market_ret * 100 
    
    # 4. EGARCH(1,1,1) Volatility
    try:
        model = arch_model(market_ret_scaled, vol='EGARCH', p=1, o=1, q=1, dist='t')
        res = model.fit(update_freq=0, disp='off')
        egarch_vol = (res.conditional_volatility / 100) * np.sqrt(252) # Trả lại scale cũ
    except Exception as e:
        st.error(f"❌ Lỗi khi chạy mô hình EGARCH: {e}")
        st.stop()
        
    # ... (Phần code tính Skewness, Correlation và CSV bên dưới giữ nguyên) ...
    
    # 2. Rolling Skewness (60 ngày)
    rolling_skew = market_ret.rolling(window=60).skew().ffill()
    
    # 3. Upside & Downside Correlation
    market_down = market_ret.where(market_ret < 0, np.nan)
    market_up = market_ret.where(market_ret > 0, np.nan)
    
    stock_down = stocks_ret.where(market_ret < 0, np.nan)
    stock_up = stocks_ret.where(market_ret > 0, np.nan)
    
    downside_corr_all = stock_down.rolling(window=60, min_periods=15).corr(market_down)
    upside_corr_all = stock_up.rolling(window=60, min_periods=15).corr(market_up)
    
    downside_corr = downside_corr_all.median(axis=1).ffill()
    upside_corr = upside_corr_all.median(axis=1).ffill()
    
    # 4. Cross-Sectional Volatility (Idiosyncratic Risk)
    roll_var_m = market_ret.rolling(window=60).var()
    roll_cov = stocks_ret.rolling(window=60).cov(market_ret)
    roll_beta = roll_cov.div(roll_var_m, axis=0)
    roll_var_i = stocks_ret.rolling(window=60).var()
    
    idio_variance = roll_var_i - (roll_beta ** 2).multiply(roll_var_m, axis=0)
    idio_variance = idio_variance.clip(lower=0)
    csv_index = idio_variance.median(axis=1).ffill()
    
    # Gom dữ liệu thành DataFrame
    metrics_df = pd.DataFrame({
        'Market_Return': market_ret,
        'EGARCH_Vol': egarch_vol,
        'Skewness': rolling_skew,
        'Downside_Corr': downside_corr,
        'Upside_Corr': upside_corr,
        'CSV_Index': csv_index
    }).dropna()
    
    return metrics_df

# ================= HÀM CHẤM ĐIỂM RỦI RO (RISK SCORING) =================
def calculate_risk_score(metrics_df):
    df = metrics_df.copy()
    scaler = MinMaxScaler()
    
    # Chuẩn hóa các biến về thang 0-1 (dựa trên dữ liệu quá khứ)
    df['Vol_Norm'] = scaler.fit_transform(df[['EGARCH_Vol']])
    df['Down_Corr_Norm'] = scaler.fit_transform(df[['Downside_Corr']])
    df['Up_Corr_Norm'] = scaler.fit_transform(df[['Upside_Corr']])
    
    # Clip Skewness để giới hạn nhiễu
    df['Skew_Clipped'] = df['Skewness'].clip(-1, 1)
    
    scores = []
    for idx, row in df.iterrows():
        vol = row['Vol_Norm']
        skew = row['Skew_Clipped']
        down_corr = row['Down_Corr_Norm']
        up_corr = row['Up_Corr_Norm']
        
        # Base Score là 50 (Vùng trung tính / Stock Picking)
        # Lực kéo về Hoảng loạn (Panic Pull)
        panic_pull = vol * down_corr * abs(min(skew, 0))  # Càng lớn càng hoảng loạn
        
        # Lực đẩy lên Hưng phấn (FOMO Push)
        fomo_push = vol * up_corr * max(skew, 0)          # Càng lớn càng hưng phấn
        
        if panic_pull > fomo_push:
            score = 50 - (50 * panic_pull)  # Kéo điểm về sát 0
        else:
            score = 50 + (50 * fomo_push)   # Đẩy điểm lên sát 100
            
        scores.append(score)
        
    df['Risk_Score'] = scores
    
    # Làm mượt đường điểm số bằng EWMA 3 ngày
    df['Risk_Score'] = df['Risk_Score'].ewm(span=3, adjust=False).mean()
    return df

# ================= GIAO DIỆN STREAMLIT =================
st.sidebar.title("Cài đặt Hệ thống")

if st.sidebar.button("🔄 Cập nhật Dữ liệu EOD", use_container_width=True):
    # 1. Xóa Cache trên RAM của Streamlit
    st.cache_data.clear()
    
    # 2. Tiêu hủy file Cache cứng trên ổ đĩa
    cache_file = "quant_risk_cache.csv"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        
    # 3. Báo cáo và tự động Refresh lại trang web
    st.sidebar.success("Đã xóa dữ liệu cũ! Đang tiến hành tải lại từ API...")
    time.sleep(1) # Dừng 1 giây cho bạn kịp đọc thông báo
    st.rerun() # Ép Streamlit tải lại toàn bộ app từ đầu

st.title("🎯 VN-Index Fear & Greed Score")
st.markdown("*Nhận diện dòng tiền Hoảng loạn (Panic Sell), Hưng phấn (FOMO) và Môi trường Stock Picking.*")

# Đọc danh sách 200 mã
try:
    df_tickers = pd.read_csv('danh_sach_200_ma.csv')
    tickers = df_tickers['Ticker'].tolist()
except FileNotFoundError:
    st.error("Không tìm thấy file danh_sach_200_ma.csv")
    st.stop()

with st.spinner('Đang tải dữ liệu và chạy mô hình EGARCH...'):
    df_vnindex, df_stocks = load_data(tickers)

if df_vnindex is not None and df_stocks is not None:
    # Tính toán
    metrics_df = calculate_quant_metrics(df_vnindex, df_stocks)
    scored_df = calculate_risk_score(metrics_df)
    
    latest = scored_df.iloc[-1]
    current_score = latest['Risk_Score']
    
    # --- ĐOẠN BỔ SUNG: Lấy ngày giao dịch gần nhất ---
    # Lấy index (ngày tháng) của dòng cuối cùng và định dạng lại thành DD/MM/YYYY
    latest_date_str = scored_df.index[-1].strftime("%d/%m/%Y")
    
    # --- PHẦN 1: ĐỒNG HỒ ĐO ĐIỂM RỦI RO (GAUGE CHART & TREND) ---
    latest_date_str = scored_df.index[-1].strftime("%d/%m/%Y")
    st.markdown(f"### 🚦Fear & Greed Score (Cập nhật chốt phiên: **{latest_date_str}**)")
    
    col1, col2 = st.columns([1, 2])
    
    # 1. Trích xuất điểm số hiện tại và điểm số hôm qua
    current_score = scored_df['Risk_Score'].iloc[-1]
    previous_score = scored_df['Risk_Score'].iloc[-2]
    
    # Tính độ lệch (Delta)
    score_change = current_score - previous_score
    
    # 2. Vẽ Đồng hồ (Gauge Chart)
    with col1:
        # Giữ nguyên logic xác định trạng thái và màu sắc
        if current_score <= 20:
            status, color, desc = "EXTREME PANIC", "red", "Hoảng loạn tột độ. Margin call."
        elif current_score >= 80:
            status, color, desc = "EXTREME FOMO", "green", "Hưng phấn quá đà. Bóng bóng ngắn hạn."
        elif 40 <= current_score <= 60:
            status, color, desc = "STOCK PICKING", "gray", "Vùng cân bằng. Tập trung tìm Alpha."
        else:
            status, color, desc = "TRANSITION", "orange", "Chuyển pha. Động lượng đang thay đổi."
            
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
        
    # 3. Hiển thị các thông số thành phần kèm Mũi tên xu hướng (Delta)
    with col2:
        st.markdown(f"#### 🔍 Trạng thái Dòng tiền (End of Day):")
        
        # Hàng 1: Điểm số tổng hợp (Risk Score)
        st.metric(
            label="🎯 Điểm Rủi Ro Hiện Tại (Risk Score)", 
            value=f"{current_score:.1f} / 100", 
            delta=f"{score_change:.1f} điểm so với hôm qua",
            # delta_color="normal" nghĩa là tăng = xanh, giảm = đỏ.
            # Tùy logic của bạn, điểm rủi ro tăng lên 100 (FOMO) là Tốt hay Xấu?
            # Ở đây tôi quy ước Tăng (hướng về FOMO) là màu Xanh (dòng tiền vào), Giảm (hướng về Panic) là màu Đỏ
            delta_color="normal" 
        )
        
        st.markdown("---") # Đường gạch ngang phân cách
        
        # Lấy giá trị thành phần của hôm qua để tính Delta
        prev = scored_df.iloc[-2]
        
        # Hàng 2: Các biến số định lượng
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("EGARCH Volatility", f"{latest['EGARCH_Vol']:.2%}", f"{(latest['EGARCH_Vol'] - prev['EGARCH_Vol']):.2%}", delta_color="inverse") # Volatility tăng là xấu (màu đỏ)
        m2.metric("Rolling Skewness", f"{latest['Skewness']:.2f}", f"{(latest['Skewness'] - prev['Skewness']):.2f}")
        m3.metric("Downside Corr", f"{latest['Downside_Corr']:.2f}", f"{(latest['Downside_Corr'] - prev['Downside_Corr']):.2f}", delta_color="inverse") # Downside Corr tăng là hoảng loạn (màu đỏ)
        m4.metric("Upside Corr", f"{latest['Upside_Corr']:.2f}", f"{(latest['Upside_Corr'] - prev['Upside_Corr']):.2f}")
        
        st.info("""
        *💡 **Cách hệ thống chấm điểm:** Kết hợp sức mạnh của 3 chiều không gian. 
        Nếu Volatility vọt lên cao + Skewness âm nặng + Downside Corr bám sát 1 => Hệ thống sẽ trừ điểm mạnh về mốc 0 (Panic). 
        Ngược lại, nếu Skewness dương và Upside Corr cao => Kéo điểm lên 100 (FOMO).*
        """)

    # --- PHẦN 2: TRỰC QUAN HÓA BỘ CHỈ BÁO THÀNH PHẦN ---
    st.markdown("### 📈 Biểu đồ Phân tích Định hướng Dòng tiền")
    
    # 1. Khai báo lưới 3 hàng, bật tính năng trục Y phụ (secondary_y) cho hàng 1 và hàng 2
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        specs=[[{"secondary_y": True}], 
               [{"secondary_y": True}], 
               [{"secondary_y": False}]],
        subplot_titles=('1. VN-Index vs EGARCH Volatility (Systemic Risk)', 
                        '2. Định hướng rủi ro (Skewness & Correlation)', 
                        '3. Cross-Sectional Volatility (%) - Cơ hội Stock Picking')
    )

    # Subplot 1: Returns (Trục chính) & EGARCH Volatility (Trục phụ)
    colors = ['seagreen' if val > 0 else 'crimson' for val in scored_df['Market_Return']]
    fig.add_trace(go.Bar(x=scored_df.index, y=scored_df['Market_Return'], name='VNINDEX Return', marker_color=colors, opacity=0.6), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=scored_df.index, y=scored_df['EGARCH_Vol'], name='EGARCH Volatility', line=dict(color='purple', width=2)), row=1, col=1, secondary_y=True)
    
    # Subplot 2: Correlations (Trục chính) & Skewness (Trục phụ)
    fig.add_trace(go.Scatter(x=scored_df.index, y=scored_df['Downside_Corr'], name='Downside Corr', line=dict(color='red', width=1.5)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=scored_df.index, y=scored_df['Upside_Corr'], name='Upside Corr', line=dict(color='green', width=1.5)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=scored_df.index, y=scored_df['Skewness'], name='Rolling Skewness', line=dict(color='black', dash='dot')), row=2, col=1, secondary_y=True)

    # Subplot 3: Cross Sectional Volatility (CSV) 
    # MẸO: Chuyển Phương sai (Variance) thành Volatility (%) để dễ nhìn
    csv_vol_percent = np.sqrt(scored_df['CSV_Index']) * 100 
    fig.add_trace(go.Scatter(x=scored_df.index, y=csv_vol_percent, name='CSV (%)', line=dict(color='royalblue', width=2), fill='tozeroy'), row=3, col=1)

    # ================= ĐỊNH DẠNG LẠI TẤT CẢ CÁC TRỤC =================
    fig.update_layout(
        height=850,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Định dạng trục Y cho từng khung
    fig.update_yaxes(title_text="Daily Return", tickformat=".1%", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="EGARCH Vol", tickformat=".1%", row=1, col=1, secondary_y=True)
    
    fig.update_yaxes(title_text="Correlation (-1 to 1)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Skewness", row=2, col=1, secondary_y=True)
    
    fig.update_yaxes(title_text="CSV Volatility (%)", tickformat=".1f", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

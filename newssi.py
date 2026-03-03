import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import datetime
import time
import os
import plotly.express as px

# --- CẤU HÌNH API KEY VNSTOCK ---
os.environ['VNSTOCK_API_KEY'] = 'vnstock_17b56a86b930db526e25e8de447a0bfd'
try:
    from vnstock import Quote
except ImportError:
    st.error("Chưa cài đặt thư viện vnstock. Hãy chạy: pip install -U vnstock")
    st.stop()

# ================= CẤU HÌNH TRANG =================
st.set_page_config(page_title="VN30 Systemic Risk Monitor", layout="wide")
PCA_WINDOW = 60
VN30_LIST = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
    'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]

# ================= HÀM TẢI & TÍNH TOÁN DỮ LIỆU LÕI =================
@st.cache_data(show_spinner=False)
def fetch_and_calculate_core_data(start_date_str, end_date_str):
    status_text = st.empty()
    progress_bar = st.progress(0)
    try:
        status_text.info(f"Đang tải VN30 Index từ {start_date_str} đến {end_date_str}...")
        quote_idx = Quote(symbol='VN30', source='KBS')
        df_idx = quote_idx.history(start=start_date_str, end=end_date_str, interval='1D')
        if df_idx is None or df_idx.empty:
            status_text.error("Không lấy được dữ liệu Index từ VCI.")
            return None, None
            
        df_idx.columns = df_idx.columns.str.lower()
        df_idx['time'] = pd.to_datetime(df_idx['time'])
        df_idx = df_idx.set_index('time').sort_index()

        all_stocks_data = []
        total_stocks = len(VN30_LIST)
        for i, symbol in enumerate(VN30_LIST):
            status_text.info(f"Đang tải mã {symbol} ({i+1}/{total_stocks})...")
            progress_bar.progress((i + 1) / total_stocks)
            try:
                quote = Quote(symbol=symbol, source='KBS')
                df = quote.history(start=start_date_str, end=end_date_str, interval='1D')
                if df is not None and not df.empty:
                    df.columns = df.columns.str.lower()
                    df['ticker'] = symbol
                    all_stocks_data.append(df[['time', 'ticker', 'close', 'volume']])
                time.sleep(1.2) 
            except: pass 
                
        status_text.info("Đang xử lý thuật toán...")
        df_stocks = pd.concat(all_stocks_data)
        df_stocks['time'] = pd.to_datetime(df_stocks['time'])
        stock_prices = df_stocks.pivot_table(index='time', columns='ticker', values='close')
        stock_rets = stock_prices.pct_change()
        
        common_index = df_idx.index.intersection(stock_rets.index)
        df_idx = df_idx.loc[common_index]
        stock_rets = stock_rets.loc[common_index]
        
        # Pillars
        idx_ret = df_idx['close'].pct_change()
        s_vol = idx_ret.rolling(window=20).std() * np.sqrt(252)
        s_lev = (df_idx['volume'] * (idx_ret < 0).astype(int)).rolling(window=5).sum() / df_idx['volume'].rolling(window=5).sum()

        s_cor_values = []
        for i in range(len(stock_rets)):
            if i < PCA_WINDOW: s_cor_values.append(np.nan)
            else:
                try:
                    pca = PCA(n_components=1)
                    pca.fit(stock_rets.iloc[i-PCA_WINDOW : i].fillna(0))
                    s_cor_values.append(pca.explained_variance_ratio_[0])
                except: s_cor_values.append(np.nan)
        s_cor = pd.Series(s_cor_values, index=stock_rets.index)

        df_stocks_aligned = df_stocks[df_stocks['time'].isin(common_index)].copy()
        df_stocks_aligned['ret_abs'] = df_stocks_aligned.groupby('ticker')['close'].pct_change().abs()
        df_stocks_aligned['amihud'] = df_stocks_aligned['ret_abs'] / (df_stocks_aligned['close'] * df_stocks_aligned['volume']).replace(0, np.nan)
        s_liq = df_stocks_aligned.groupby('time')['amihud'].mean().rolling(window=20).mean().reindex(common_index)
        
        base_pillars = pd.DataFrame({'S_VOL': s_vol, 'S_LEV': s_lev, 'S_COR': s_cor, 'S_LIQ': s_liq})
        status_text.empty(); progress_bar.empty()
        return df_idx, base_pillars
    except Exception as e:
        status_text.error(f"Lỗi hệ thống: {e}"); return None, None

# ================= HÀM TỔNG HỢP SSI =================
def calculate_ssi_dynamic(df_idx, base_pillars, bond_input, ma_period):
    if isinstance(bond_input, pd.Series):
        bond_aligned = bond_input.reindex(df_idx.index, method='ffill').bfill()
        s_val = -( (1 / (df_idx['close'] / 100)) - bond_aligned )
    else:
        s_val = -( (1 / (df_idx['close'] / 100)) - bond_input )
        
    df_metrics = base_pillars.copy()
    df_metrics['S_VAL'] = s_val
    df_metrics = df_metrics.dropna()
    if df_metrics.empty: return None, None
        
    df_rank = df_metrics.rank(pct=True)
    pca_final = PCA(n_components=1).fit(df_rank)
    weights = np.abs(pca_final.components_[0])
    weights /= weights.sum()
    
    df_metrics['SSI_Index'] = (df_rank * weights).sum(axis=1)
    df_metrics['VN30_Close'] = df_idx['close']
    df_metrics[f'MA{ma_period}'] = df_idx['close'].rolling(window=ma_period).mean()
    
    return df_metrics, weights

# ================= HÀM TẠO PDF =================
def generate_pdf_report(df_metrics, weights, status, status_color_name, last_ssi, last_date, ma_period):
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.suptitle(f"VN30 ESR MONITOR - {last_date}", fontsize=16, weight='bold', color='navy')
        
        ax_text = fig.add_axes([0.05, 0.75, 0.4, 0.15]); ax_text.axis('off')
        summary_text = f"STATUS: {status}\nSSI: {last_ssi:.1%}\nVN30: {df_metrics['VN30_Close'].iloc[-1]:.2f}"
        ax_text.text(0, 0.5, summary_text, fontsize=14, color=status_color_name, weight='bold', va='center')

        ax_w = fig.add_axes([0.55, 0.70, 0.4, 0.2])
        ax_w.bar(df_metrics.columns[:5], weights, color='steelblue', alpha=0.7)
        ax_w.set_title("Risk Contribution (PCA Weights)", fontsize=10)

        ax_main = fig.add_axes([0.05, 0.05, 0.9, 0.55])
        ax_main.plot(df_metrics.index, df_metrics['SSI_Index'], color='darkred', label='SSI (Left)')
        ax_main.set_ylabel("SSI Index", color='darkred')
        
        ax_main2 = ax_main.twinx()
        ax_main2.plot(df_metrics.index, df_metrics['VN30_Close'], color='steelblue', alpha=0.6, label='VN30 (Right)')
        ax_main2.plot(df_metrics.index, df_metrics[f'MA{ma_period}'], color='orange', linewidth=1, label=f'MA{ma_period} (Right)')
        ax_main2.set_ylabel("VN30 Points", color='steelblue')
        
        ax_main.legend(loc='upper left'); ax_main2.legend(loc='upper right')
        ax_main.grid(True, alpha=0.3)
        pdf.savefig(fig); plt.close()

    return pdf_buffer.getvalue()

# ================= UI STREAMLIT =================
st.sidebar.title("Tham số Hệ thống")
mode = st.sidebar.radio("Chọn chế độ:", ["A) Đo lường hiện tại", "B) Backtest"])
ma_choice = st.sidebar.selectbox("Chọn chu kỳ MA cho VN30:", [20, 60, 125, 252], index=1)

if mode == "A) Đo lường hiện tại":
    st.title("📊 VN30 ESR Monitor")
    bond_yield_input = st.sidebar.number_input("Bond Yield hiện tại (%)", value=4.20, step=0.1) / 100
    end_date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date_str = (datetime.datetime.now() - datetime.timedelta(days=500)).strftime('%Y-%m-%d')
    
    df_idx, base_pillars = fetch_and_calculate_core_data(start_date_str, end_date_str)
    if df_idx is not None:
        df_metrics, weights = calculate_ssi_dynamic(df_idx, base_pillars, bond_yield_input, ma_choice)
        if df_metrics is not None:
            last_ssi = df_metrics['SSI_Index'].iloc[-1]
            status = "SAFE" if last_ssi < 0.5 else ("WARNING" if last_ssi < 0.8 else "CRITICAL")
            color = "green" if status == "SAFE" else ("orange" if status == "WARNING" else "red")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"<div style='padding:20px; border-radius:10px; background:#f8f9fa; border-left:5px solid {color};'><h3>{status}</h3><h1>{last_ssi:.1%}</h1></div>", unsafe_allow_html=True)
                pdf_bytes = generate_pdf_report(df_metrics, weights, status, color, last_ssi, df_metrics.index[-1].date(), ma_choice)
                st.download_button("📥 Tải Báo Cáo PDF", pdf_bytes, file_name="ESR_Report.pdf", use_container_width=True)

            with col2:
                fig_bar = px.bar(x=df_metrics.columns[:5], y=weights, title="PCA Weights")
                st.plotly_chart(fig_bar, use_container_width=True)

            fig_main = make_subplots(specs=[[{"secondary_y": True}]])
            fig_main.add_trace(go.Scatter(x=df_metrics.index, y=df_metrics['SSI_Index'], name='SSI', line=dict(color='darkred')), secondary_y=False)
            fig_main.add_trace(go.Scatter(x=df_metrics.index, y=df_metrics['VN30_Close'], name='VN30', line=dict(color='steelblue', width=1, dash='dot')), secondary_y=True)
            fig_main.add_trace(go.Scatter(x=df_metrics.index, y=df_metrics[f'MA{ma_choice}'], name=f'MA{ma_choice}', line=dict(color='orange', width=1.5)), secondary_y=True)
            fig_main.update_layout(height=500, hovermode="x unified")
            st.plotly_chart(fig_main, use_container_width=True)

# ================= TRONG MỤC GIAO DIỆN BACKTEST =================

else:
    st.title("📈 VN30 ESR - Backtest Mode")
    st.sidebar.markdown("---")
    
    # 1. Khởi tạo session state để lưu trữ kết quả nếu chưa có
    if 'bt_df_metrics' not in st.session_state:
        st.session_state.bt_df_metrics = None

    u_start = st.sidebar.date_input("Bắt đầu", datetime.date(2022, 1, 1))
    u_end = st.sidebar.date_input("Kết thúc", datetime.date(2022, 12, 31))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**DỮ LIỆU BOND YIELD (TÙY CHỌN)**")
    bond_file = st.sidebar.file_uploader("Tải lên file lịch sử (CSV)", type=['csv'])
    st.sidebar.caption("💡 Ghi chú: Tải dữ liệu trái phiếu 10 năm tương ứng khoảng thời gian trên trang investing.com")
    
    fixed_bond = st.sidebar.number_input("Bond Yield giả định cố định (%)", value=4.20) / 100

    # 2. Nút bấm chỉ thực hiện việc TẢI và TÍNH TOÁN LÕI
    if st.sidebar.button("🚀 Chạy Backtest"):
        # Cố định f_start ở mức đệm tối đa (252 ngày MA + 60 ngày PCA) 
        # để giữ cache key không đổi khi thay đổi MA dropdown
        f_start = (u_start - datetime.timedelta(days=320)).strftime('%Y-%m-%d')
        f_end = u_end.strftime('%Y-%m-%d')
        
        df_idx, base_pillars = fetch_and_calculate_core_data(f_start, f_end)
        
        if df_idx is not None:
            bond_input = fixed_bond
            if bond_file:
                try:
                    db = pd.read_csv(bond_file)
                    db['Ngày'] = pd.to_datetime(db['Ngày'], format='%d/%m/%Y', errors='coerce')
                    bond_input = db.dropna(subset=['Ngày']).set_index('Ngày').sort_index()['Lần cuối'] / 100
                except: pass
            
            # Tính toán SSI và lưu vào session_state (chưa tính MA ở đây để linh hoạt)
            # Chúng ta truyền ma_choice mặc định là 252 để đảm bảo tính sẵn cho mọi trường hợp
            df_res, _ = calculate_ssi_dynamic(df_idx, base_pillars, bond_input, 252)
            st.session_state.bt_df_metrics = df_res
            st.success("Đã tải xong dữ liệu!")

    # 3. Hiển thị biểu đồ: Tự động cập nhật khi ma_choice thay đổi mà không tải lại data
    if st.session_state.bt_df_metrics is not None:
        df_plot_full = st.session_state.bt_df_metrics.copy()
        
        # Tính toán đường MA mới dựa trên lựa chọn hiện tại của người dùng
        # Việc tính MA trên DataFrame có sẵn cực nhanh, không tốn tài nguyên
        df_plot_full[f'MA{ma_choice}'] = df_plot_full['VN30_Close'].rolling(window=ma_choice).mean()
        
        # Lọc đúng khoảng thời gian hiển thị
        df_filtered = df_plot_full.loc[u_start:u_end]
        
        st.subheader(f"Kết quả Backtest ({u_start} đến {u_end})")
        
        fig_bt = make_subplots(specs=[[{"secondary_y": True}]])
        
        # SSI (Trục trái)
        fig_bt.add_trace(
            go.Scatter(x=df_filtered.index, y=df_filtered['SSI_Index'], name='SSI', line=dict(color='darkred')),
            secondary_y=False,
        )
        # VN30 (Trục phải)
        fig_bt.add_trace(
            go.Scatter(x=df_filtered.index, y=df_filtered['VN30_Close'], name='VN30', line=dict(color='steelblue', width=1, dash='dot')),
            secondary_y=True,
        )
        # MA được chọn (Trục phải)
        fig_bt.add_trace(
            go.Scatter(x=df_filtered.index, y=df_filtered[f'MA{ma_choice}'], name=f'MA{ma_choice}', line=dict(color='orange', width=1.5)),
            secondary_y=True,
        )

        fig_bt.update_layout(height=600, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
        fig_bt.update_yaxes(title_text="SSI Index", secondary_y=False)
        fig_bt.update_yaxes(title_text="VN30 Points", secondary_y=True)
        
        st.plotly_chart(fig_bt, use_container_width=True)



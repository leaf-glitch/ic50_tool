import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io

# 設定網頁標題與排版
st.set_page_config(page_title="Dose-Response IC50 Tool", layout="centered")
st.title("📊 智慧定位型 IC50 曲線擬合工具")
st.write("本系統已優化：支援無標題欄位、任意欄位起點之 Excel 數據。")

# 1. 定義四參數邏輯斯模型 (4PL Model)
def log_4pl(x, min_response, max_response, ic50, hill_slope):
    return min_response + (max_response - min_response) / (1 + (x / ic50) ** hill_slope)

# 2. 網頁檔案上傳元件
uploaded_file = st.file_uploader("📂 請選擇你的 Excel 檔案 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # 💡 【核心修正 1】讀取時「不」把第一列當標題，強制用數字 0, 1, 2... 作為欄位名
        raw_df = pd.read_excel(uploaded_file, header=None)
        
        # 將欄位名稱改成對應 Excel 的英文字母 A, B, C, D... 方便使用者辨識
        import string
        num_cols = raw_df.shape[1]
        alphabet_cols = []
        for i in range(num_cols):
            # 支援超過 26 欄的情況
            if i < 26:
                alphabet_cols.append(string.ascii_uppercase[i])
            else:
                alphabet_cols.append(string.ascii_uppercase[i//26 - 1] + string.ascii_uppercase[i%26])
        raw_df.columns = alphabet_cols

        # 移除完全空白的橫列與直欄
        raw_df.dropna(how='all', inplace=True)
        raw_df.dropna(how='all', axis=1, inplace=True)

        st.write("---")
        st.subheader("⚙️ 欄位設定")
        
        all_columns = raw_df.columns.tolist()
        
        col_select1, col_select2 = st.columns(2)
        
        with col_select1:
            # 依據你的截圖，預設幫你找 D 欄，找不到就選第一個
            default_x_idx = all_columns.index('D') if 'D' in all_columns else 0
            concentration_col = st.selectbox("🎯 請選擇【濃度】所在的英文字母欄：", all_columns, index=default_x_idx)
            
        with col_select2:
            remaining_cols = [c for c in all_columns if c != concentration_col]
            # 依據你的截圖，預設幫你勾選 E, F, G 欄
            default_y = [c for c in ['E', 'F', 'G'] if c in remaining_cols]
            replicate_cols = st.multiselect("🧪 請勾選【重複實驗數據】所在的欄位（可多選）：", remaining_cols, default=default_y if default_y else remaining_cols[:3])

        if not replicate_cols:
            st.warning("⚠️ 請至少勾選一個重複實驗數據欄位。")
            st.stop()

        # 根據使用者的選擇，提取數據
        selected_cols = [concentration_col] + replicate_cols
        df_clean = raw_df[selected_cols].copy()
        
        # 強制轉換成數字，此時會自動把上方或下方的空白與文字轉成 NaN
        df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
        
        # 💡 【核心修正 2】剔除含有空值的橫列。這樣就會精準只留下 D, E, F, G 同時有數字的那 8 列！
        df_clean.dropna(inplace=True)

        # 【數據預覽區塊】顯示成功去空值後的純數字
        with st.expander("📄 點擊展開/收合：目前選定純數字數據預覽"):
            st.dataframe(df_clean, use_container_width=True)

        if df_clean.empty:
            st.error("❌ 錯誤：所選欄位轉為數字並清洗空值後，已無可用數據。請確認你勾選的欄位是否正確。")
            st.stop()

        # 3. 提取濃度與重複實驗數據
        raw_concentrations = df_clean[concentration_col].values
        raw_replicates = df_clean[replicate_cols].values
        replicate_count = len(replicate_cols)

        # 尋找濃度為 0 的控制組並進行基準計算
        control_idx = np.argmin(np.abs(raw_concentrations))
        control_conc_value = raw_concentrations[control_idx]
        control_group_mean = raw_replicates[control_idx].mean()

        st.success(f"📊 欄位解析成功！偵測到控制組為【欄位 {concentration_col} = {control_conc_value}】，原始平均值：{control_group_mean:.2f} (n={replicate_count})")

        # 進行歸一化
        normalized_replicates = (raw_replicates / control_group_mean) * 100
        mean_responses = normalized_replicates.mean(axis=1)
        std_errors = normalized_replicates.std(axis=1)

        # 對數修正
        concentrations = np.where(raw_concentrations <= 0, 1e-6, raw_concentrations)
        std_errors = np.nan_to_num(std_errors, nan=1e-5)
        std_errors = np.where(std_errors == 0, 1e-5, std_errors)

        # 4. 進行曲線擬合 (Curve Fitting)
        initial_guess = [min(mean_responses), max(mean_responses), np.median(concentrations), 1.0]
        bounds = ([0, 0, concentrations.min()*0.1, 0.1], [50, 250, concentrations.max()*10, 5])

        popt, pcov = curve_fit(log_4pl, concentrations, mean_responses, p0=initial_guess, bounds=bounds, sigma=std_errors)
        fitted_min, fitted_max, fitted_ic50, fitted_slope = popt
        ic50_error = np.sqrt(np.diag(pcov))[2]
        
        # 網頁數字面板
        st.subheader("📈 IC50 擬合結果")
        col1, col2, col3 = st.columns(3)
        col1.metric("💡 IC50 推估值", f"{fitted_ic50:.4f} ± {ic50_error:.4f}")
        col2.metric("Top (最高相對反應)", f"{fitted_max:.2f}%")
        col3.metric("Bottom (最低相對反應)", f"{fitted_min:.2f}%")
        
    except Exception as e:
        st.error(f"❌ 數據處理或擬合失敗。原因可能是數值未呈現 S 型趨勢。錯誤訊息: {e}")
        st.stop()

    # 5. 繪圖
    fig, ax = plt.subplots(figsize=(8, 6))

    # A. 數據點與 Error Bar
    ax.errorbar(concentrations, mean_responses, yerr=std_errors, fmt='o', color='black', markeredgecolor='black', markerfacecolor='white', markersize=7, capsize=5, label=f'Normalized Data (Mean ± SD, n={replicate_count})')

    # B. 個別原始數據點
    for i in range(replicate_count):
        ax.scatter(concentrations, normalized_replicates[:, i], color='lightgray', alpha=0.5, s=20, zorder=2)

    # C. 產生平滑的 X 軸數據來繪製 4PL 擬合曲線
    x_smooth = np.logspace(np.log10(concentrations.min()), np.log10(concentrations.max()), 500)
    y_smooth = log_4pl(x_smooth, *popt)

    # 畫出黑色擬合曲線
    ax.plot(x_smooth, y_smooth, '-', color='black', linewidth=2, label='4PL Fitted Curve', zorder=1)

    # D. 標註 IC50 位置
    ax.axvline(x=fitted_ic50, color='dimgray', linestyle='--', alpha=0.6)
    ax.axhline(y=(fitted_max + fitted_min)/2, color='dimgray', linestyle='--', alpha=0.6)
    ax.text(fitted_ic50 * 1.3, (fitted_max + fitted_min)/2 + (fitted_max - fitted_min)*0.05, f'IC50 = {fitted_ic50:.4f}', color='black', fontsize=12, weight='bold')

    # E. 調整軸標籤
    ax.set_xscale('log')
    ax.set_xlabel('Concentration (Log Scale)', fontsize=12)
    ax.set_ylabel('Cell Viability (% of Control)', fontsize=12)
    ax.set_title('Dose-Response Curve (Normalized to Control)', fontsize=14, weight='bold')
    ax.legend(loc='best')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()

    # 秀出圖表在網頁上
    st.pyplot(fig)

    # 💾 提供高解析度圖片下載按鈕
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300)
    img_buffer.seek(0)
    
    st.download_button(
        label="💾 下載高解析度圖表 (PNG)",
        data=img_buffer,
        file_name="ic50_curve.png",
        mime="image/png"
    )
    plt.close()

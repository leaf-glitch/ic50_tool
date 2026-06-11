import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io

# 設定網頁標題與排版
st.set_page_config(page_title="Dose-Response IC50 Tool", layout="centered")
st.title("📊 智慧型 IC50 曲線擬合工具")
st.write("上傳 Excel 檔案，系統將自動以「濃度為 0 的組別」為 100% 基準點進行歸一化分析。")

# 1. 定義四參數邏輯斯模型 (4PL Model)
def log_4pl(x, min_response, max_response, ic50, hill_slope):
    return min_response + (max_response - min_response) / (1 + (x / ic50) ** hill_slope)

# 2. 網頁檔案上傳元件
uploaded_file = st.file_uploader("📂 請拖放或選擇你的 Excel 檔案 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # 自動尋找真正的資料起點 (改用 uploaded_file 讀取)
        raw_df = pd.read_excel(uploaded_file, header=None)

        # 移除完全空白的行與列
        raw_df.dropna(how='all', inplace=True)
        raw_df.dropna(how='all', axis=1, inplace=True)

        # 尋找第一個看起來像數字資料的起點
        valid_rows = []
        for idx, val in enumerate(raw_df.iloc[:, 0]):
            try:
                float_val = float(val)
                if not np.isnan(float_val):
                    valid_rows.append(idx)
            except ValueError:
                continue

        if not valid_rows:
            st.error("❌ 錯誤：在 Excel 的第一欄中找不到任何有效的數字（濃度）。請檢查 Excel 格式。")
            st.stop()

        # 根據找到的有效列，重新建立乾淨的 DataFrame
        df = raw_df.iloc[valid_rows].copy()

        # 將所有欄位強制轉換成數字
        df = df.apply(pd.to_numeric, errors='coerce')

        # 剔除含有任何 NaN 的橫列
        df.dropna(inplace=True)

        if df.empty:
            st.error("❌ 錯誤：清洗完空值後，已經沒有可用的完整數據，請檢查 Excel 內是否有很多空白格子。")
            st.stop()

        # 3. 提取濃度與重複實驗數據
        raw_concentrations = df.iloc[:, 0].values  
        raw_replicates = df.iloc[:, 1:].values
        replicate_count = raw_replicates.shape[1]

        # 尋找濃度為 0 的控制組
        control_idx = np.argmin(np.abs(raw_concentrations))
        control_conc_value = raw_concentrations[control_idx]

        # 算出「濃度為 0 組」的重複實驗平均值
        control_group_mean = raw_replicates[control_idx].mean()

        # 將原本終端機的文字顯示在網頁上
        st.success(f"📊 數據自動清洗與歸一化成功！偵測到控制組濃度為：{control_conc_value}，原始平均值：{control_group_mean:.2f} (n={replicate_count})")

        # 進行歸一化
        normalized_replicates = (raw_replicates / control_group_mean) * 100

        # 重新計算歸一化後的各組平均值與標準差
        mean_responses = normalized_replicates.mean(axis=1)
        std_errors = normalized_replicates.std(axis=1)

        # 对数修正
        concentrations = np.where(raw_concentrations <= 0, 1e-6, raw_concentrations)

        # 防錯機制
        std_errors = np.nan_to_num(std_errors, nan=1e-5)
        std_errors = np.where(std_errors == 0, 1e-5, std_errors)

        # 4. 進行曲線擬合 (Curve Fitting)
        initial_guess = [min(mean_responses), max(mean_responses), np.median(concentrations), 1.0]
        bounds = ([0, 0, concentrations.min()*0.1, 0.1], [50, 250, concentrations.max()*10, 5])

        popt, pcov = curve_fit(log_4pl, concentrations, mean_responses, p0=initial_guess, bounds=bounds, sigma=std_errors)
        fitted_min, fitted_max, fitted_ic50, fitted_slope = popt
        ic50_error = np.sqrt(np.diag(pcov))[2]
        
        # 網頁高質感數字面板
        st.subheader("📈 IC50 擬合結果")
        col1, col2, col3 = st.columns(3)
        col1.metric("💡 IC50 推估值", f"{fitted_ic50:.4f} ± {ic50_error:.4f}")
        col2.metric("Top (最高相對反應)", f"{fitted_max:.2f}%")
        col3.metric("Bottom (最低相對反應)", f"{fitted_min:.2f}%")
        
    except Exception as e:
        st.error(f"❌ 擬合失敗或數據不符 S 型趨勢。錯誤訊息: {e}")
        st.stop()

    # 5. 繪圖與自動存檔 (改用網頁畫布 st.pyplot)
    fig, ax = plt.subplots(figsize=(8, 6))

    # A. 畫出歸一化後的數據點與 Error Bar (黑色外框、白色中心)
    ax.errorbar(concentrations, mean_responses, yerr=std_errors, fmt='o', color='black', markeredgecolor='black', markerfacecolor='white', markersize=7, capsize=5, label=f'Normalized Data (Mean ± SD, n={replicate_count})')

    # B. 畫出歸一化後的個別原始數據點 (維持半透明淺灰色)
    for i in range(replicate_count):
        ax.scatter(concentrations, normalized_replicates[:, i], color='lightgray', alpha=0.5, s=20, zorder=2)

    # C. 產生平滑 of X 軸數據來繪製 4PL 擬合曲線
    x_smooth = np.logspace(np.log10(concentrations.min()), np.log10(concentrations.max()), 500)
    y_smooth = log_4pl(x_smooth, *popt)

    # 畫出黑色擬合曲線
    ax.plot(x_smooth, y_smooth, '-', color='black', linewidth=2, label='4PL Fitted Curve', zorder=1)

    # D. 標註 IC50 位置 (輔助虛線為深灰色)
    ax.axvline(x=fitted_ic50, color='dimgray', linestyle='--', alpha=0.6)
    ax.axhline(y=(fitted_max + fitted_min)/2, color='dimgray', linestyle='--', alpha=0.6)

    # 標註純黑色文字
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

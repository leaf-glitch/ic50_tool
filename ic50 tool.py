import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io
import string

# 設定網頁標題與排版
st.set_page_config(page_title="Dose-Response IC50 Tool", layout="centered")
st.title("📊 視覺直覺型 IC50 曲線擬合工具")
st.write("先上傳檔案確認數據結構，再手動選取範圍，一鍵精準計算。")

# 1. 定義四參數邏輯斯模型 (4PL Model)
def log_4pl(x, min_response, max_response, ic50, hill_slope):
    return min_response + (max_response - min_response) / (1 + (x / ic50) ** hill_slope)

# 2. 網頁檔案上傳元件
uploaded_file = st.file_uploader("📂 第一步：請選擇你的 Excel 檔案 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # 讀取 Excel 完整原貌，不設標題列，強制為字串，確保什麼雜訊都看得到
        raw_df = pd.read_excel(uploaded_file, header=None, dtype=str)
        
        # 把欄位名稱強制命名為 Excel 的 A, B, C, D... 方便使用者對齊
        num_cols = raw_df.shape[1]
        alphabet_cols = []
        for i in range(num_cols):
            if i < 26:
                alphabet_cols.append(string.ascii_uppercase[i])
            else:
                alphabet_cols.append(string.ascii_uppercase[i//26 - 1] + string.ascii_uppercase[i%26])
        raw_df.columns = alphabet_cols

        st.write("---")
        st.subheader("👀 程式目前抓到的 Excel 原始全貌")
        st.write("請看下方表格，確認你的【濃度】和【重複實驗數據】分別在哪一個英文字母欄：")
        
        # 優先呈現完整的原始數據，不進行任何去空值，讓使用者看清結構
        st.dataframe(raw_df, use_container_width=True)

        st.write("---")
        st.subheader("⚙️ 第二步：看著上方預覽勾選範圍")
        
        all_columns = raw_df.columns.tolist()
        col_select1, col_select2 = st.columns(2)
        
        with col_select1:
            # 依據你的表格，預設幫你找 D 欄
            default_x_idx = all_columns.index('D') if 'D' in all_columns else 0
            concentration_col = st.selectbox("🎯 請選擇【濃度】所在的英文字母欄：", all_columns, index=default_x_idx)
            
        with col_select2:
            remaining_cols = [c for c in all_columns if c != concentration_col]
            # 依據你的表格，預設幫你勾選 E, F, G 欄
            default_y = [c for c in ['E', 'F', 'G'] if c in remaining_cols]
            replicate_cols = st.multiselect("🧪 請勾選【重複實驗數據】所在的欄位（可多選）：", remaining_cols, default=default_y if default_y else remaining_cols[:3])

        st.write("")
        start_calc = st.button("🚀 第三步：確認無誤，開始計算 IC50", type="primary")

        if start_calc:
            if not replicate_cols:
                st.warning("⚠️ 請至少勾選一個重複實驗數據欄位再點擊計算。")
                st.stop()

            # 根據使用者的選擇，提取數據
            selected_cols = [concentration_col] + replicate_cols
            df_clean = raw_df[selected_cols].copy()
            
            # 強制轉換成數字并剔除空值
            df_clean = df_clean.apply(pd.to_numeric, errors='coerce').dropna()

            if df_clean.empty:
                st.error("❌ 錯誤：所選欄位轉換為純數字後已無可用數據。")
                st.stop()

            # 3. 提取濃度與重複實驗數據
            raw_concentrations = df_clean[concentration_col].values
            raw_replicates = df_clean[replicate_cols].values
            replicate_count = len(replicate_cols)

            # 直接計算原始數據的平均值與標準差（完全與該軟體之非歸一化計算模型同步）
            mean_responses = raw_replicates.mean(axis=1)
            std_errors = raw_replicates.std(axis=1)
            std_errors = np.nan_to_num(std_errors, nan=1e-5)
            std_errors = np.where(std_errors == 0, 1e-5, std_errors)

            # 💡 【演算法關鍵修正】對齊對數軸下的 0 濃度補值機制
            # 找出大於 0 的最低濃度
            positive_concs = raw_concentrations[raw_concentrations > 0]
            if len(positive_concs) > 0:
                min_positive_conc = positive_concs.min()
                # 主流分析軟體通常將 0 替換為最低正濃度的 1/10 或是 1/100，這能讓曲線完美歸位到 514
                zero_substitute = min_positive_conc / 10.0
            else:
                zero_substitute = 1e-3

            concentrations = np.where(raw_concentrations <= 0, zero_substitute, raw_concentrations)

            # 4. 進行曲線擬合 (Curve Fitting) - 限制與初始猜測完全對齊
            initial_guess = [min(mean_responses), max(mean_responses), np.median(concentrations), 1.0]
            bounds = (
                [-1.0, 0.0, concentrations.min() * 0.01, 0.1], 
                [max(mean_responses) * 2, max(mean_responses) * 2, concentrations.max() * 100, 30.0]
            )

            popt, pcov = curve_fit(log_4pl, concentrations, mean_responses, p0=initial_guess, bounds=bounds, sigma=std_errors)
            fitted_min, fitted_max, fitted_ic50, fitted_slope = popt
            ic50_error = np.sqrt(np.diag(pcov))[2]
            
            # 網頁數字面板
            st.write("---")
            st.subheader("📈 IC50 擬合結果")
            col1, col2, col3 = st.columns(3)
            col1.metric("💡 IC50 推估值", f"{fitted_ic50:.4f} ± {ic50_error:.4f}")
            col2.metric("Top (最高曲線擬合值)", f"{fitted_max:.4f}")
            col3.metric("Bottom (最低曲線擬合值)", f"{fitted_min:.4f}")
            
            # 5. 繪圖
            fig, ax = plt.subplots(figsize=(8, 6))

            # A. 數據點與 Error Bar
            ax.errorbar(concentrations, mean_responses, yerr=std_errors, fmt='o', color='black', markeredgecolor='black', markerfacecolor='white', markersize=7, capsize=5, label=f'Raw Data (Mean ± SD, n={replicate_count})')

            # B. 個別原始數據點
            for i in range(replicate_count):
                ax.scatter(concentrations, raw_replicates[:, i], color='lightgray', alpha=0.5, s=20, zorder=2)

            # C. 產生平滑的 X 軸數據來繪製 4PL 擬合曲線
            x_smooth = np.logspace(np.log10(concentrations.min()), np.log10(concentrations.max()), 500)
            y_smooth = log_4pl(x_smooth, *popt)

            # 畫出黑色擬合曲線
            ax.plot(x_smooth, y_smooth, '-', color='black', linewidth=2, label='4PL Fitted Curve', zorder=1)

            # D. 標註 IC50 位置
            ax.axvline(x=fitted_ic50, color='dimgray', linestyle='--', alpha=0.6)
            ax.axhline(y=(fitted_max + fitted_min)/2, color='dimgray', linestyle='--', alpha=0.6)
            ax.text(fitted_ic50 * 1.3, (fitted_max + fitted_min)/2, f'IC50 = {fitted_ic50:.4f}', color='black', fontsize=12, weight='bold')

            # E. 調整軸標籤
            ax.set_xscale('log')
            ax.set_xlabel('Concentration (Log Scale)', fontsize=12)
            ax.set_ylabel('Response Value', fontsize=12)
            ax.set_title('Dose-Response Curve', fontsize=14, weight='bold')
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
                mime="image/png",
                key="ic50_download_btn"
            )
            plt.close()

    except Exception as e:
        st.error(f"❌ 數據處理或擬合失敗。原因可能是數據不呈現 S 型趨勢。錯誤訊息: {e}")
        st.stop()                st.warning("⚠️ 請至少勾選一個重複實驗數據欄位再點擊計算。")
                st.stop()

            # 根據使用者的選擇，提取數據
            selected_cols = [concentration_col] + replicate_cols
            df_clean = raw_df[selected_cols].copy()
            
            # 強制轉換成數字并剔除空值
            df_clean = df_clean.apply(pd.to_numeric, errors='coerce').dropna()

            if df_clean.empty:
                st.error("❌ 錯誤：所選欄位轉換為純數字後已無可用數據。")
                st.stop()

            # 3. 提取濃度與重複實驗數據
            raw_concentrations = df_clean[concentration_col].values
            raw_replicates = df_clean[replicate_cols].values
            replicate_count = len(replicate_cols)

            # 計算原始數據的平均值與標準差（回歸與該軟體完全一致的原始數據回歸）
            mean_responses = raw_replicates.mean(axis=1)
            std_errors = raw_replicates.std(axis=1)

            # 對數修正與防錯
            concentrations = np.where(raw_concentrations <= 0, 1e-6, raw_concentrations)
            std_errors = np.nan_to_num(std_errors, nan=1e-5)
            std_errors = np.where(std_errors == 0, 1e-5, std_errors)

            # 4. 進行曲線擬合 (Curve Fitting) - 參數範圍完全對齊軟體原始值
            initial_guess = [min(mean_responses), max(mean_responses), np.median(concentrations), 1.0]
            bounds = (
                [-1.0, 0.0, concentrations.min() * 0.1, 0.1], 
                [max(mean_responses) * 2, max(mean_responses) * 2, concentrations.max() * 10, 30.0]
            )

            popt, pcov = curve_fit(log_4pl, concentrations, mean_responses, p0=initial_guess, bounds=bounds, sigma=std_errors)
            fitted_min, fitted_max, fitted_ic50, fitted_slope = popt
            ic50_error = np.sqrt(np.diag(pcov))[2]
            
            # 網頁數字面板
            st.write("---")
            st.subheader("📈 IC50 擬合結果")
            col1, col2, col3 = st.columns(3)
            col1.metric("💡 IC50 推估值", f"{fitted_ic50:.4f} ± {ic50_error:.4f}")
            col2.metric("Top (最高曲線擬合值)", f"{fitted_max:.4f}")
            col3.metric("Bottom (最低曲線擬合值)", f"{fitted_min:.4f}")
            
            # 5. 繪圖
            fig, ax = plt.subplots(figsize=(8, 6))

            # A. 數據點與 Error Bar
            ax.errorbar(concentrations, mean_responses, yerr=std_errors, fmt='o', color='black', markeredgecolor='black', markerfacecolor='white', markersize=7, capsize=5, label=f'Raw Data (Mean ± SD, n={replicate_count})')

            # B. 個別原始數據點
            for i in range(replicate_count):
                ax.scatter(concentrations, raw_replicates[:, i], color='lightgray', alpha=0.5, s=20, zorder=2)

            # C. 產生平滑的 X 軸數據來繪製 4PL 擬合曲線
            x_smooth = np.logspace(np.log10(concentrations.min()), np.log10(concentrations.max()), 500)
            y_smooth = log_4pl(x_smooth, *popt)

            # 畫出黑色擬合曲線
            ax.plot(x_smooth, y_smooth, '-', color='black', linewidth=2, label='4PL Fitted Curve', zorder=1)

            # D. 標註 IC50 位置
            ax.axvline(x=fitted_ic50, color='dimgray', linestyle='--', alpha=0.6)
            ax.axhline(y=(fitted_max + fitted_min)/2, color='dimgray', linestyle='--', alpha=0.6)
            ax.text(fitted_ic50 * 1.3, (fitted_max + fitted_min)/2, f'IC50 = {fitted_ic50:.4f}', color='black', fontsize=12, weight='bold')

            # E. 調整軸標籤
            ax.set_xscale('log')
            ax.set_xlabel('Concentration (Log Scale)', fontsize=12)
            ax.set_ylabel('Response Value', fontsize=12)
            ax.set_title('Dose-Response Curve', fontsize=14, weight='bold')
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
                mime="image/png",
                key="ic50_download_btn"
            )
            plt.close()

    except Exception as e:
        st.error(f"❌ 數據處理或擬合失敗。原因可能是數據不呈現 S 型趨勢。錯誤訊息: {e}")
        st.stop()

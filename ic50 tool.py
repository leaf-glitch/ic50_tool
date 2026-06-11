import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. 定義四參數邏輯斯模型 (4PL Model)
def log_4pl(x, min_response, max_response, ic50, hill_slope):
    return min_response + (max_response - min_response) / (1 + (x / ic50) ** hill_slope)

# 2. 讀取 Excel 檔案
file_path = r'c:/Users/PC01/Desktop/TEST.xlsx'  # 👈 請確認你的 Excel 檔名是否正確

if not os.path.exists(file_path):
    print(f"❌ 錯誤：找不到檔案 '{file_path}'，請確認檔案路徑與檔名是否正確。")
    exit()

# 自動尋找真正的資料起點
raw_df = pd.read_excel(file_path, header=None)

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
    print("❌ 錯誤：在 Excel 的第一欄中找不到任何有效的數字（濃度）。請檢查 Excel 格式。")
    exit()

# 根據找到的有效列，重新建立乾淨的 DataFrame
df = raw_df.iloc[valid_rows].copy()

# 將所有欄位強制轉換成數字
df = df.apply(pd.to_numeric, errors='coerce')

# 剔除含有任何 NaN 的橫列
df.dropna(inplace=True)

if df.empty:
    print("❌ 錯誤：清洗完空值後，已經沒有可用的完整數據，請檢查 Excel 內是否有很多空白格子。")
    exit()

# 3. 提取濃度與重複實驗數據
raw_concentrations = df.iloc[:, 0].values  
raw_replicates = df.iloc[:, 1:].values
replicate_count = raw_replicates.shape[1]

# 尋找濃度為 0 的控制組
control_idx = np.argmin(np.abs(raw_concentrations))
control_conc_value = raw_concentrations[control_idx]

# 算出「濃度為 0 組」的重複實驗平均值
control_group_mean = raw_replicates[control_idx].mean()

print("====== 📊 數據自動清洗與歸一化成功 ======")
print(f"📍 偵測到控制組（濃度最接近 0）：實際數值為 {control_conc_value}")
print(f"✨ 控制組原始平均值為：{control_group_mean:.2f}（以此作為 100% 基準）")

# 進行歸一化
normalized_replicates = (raw_replicates / control_group_mean) * 100

# 重新計算歸一化後的各組平均值與標準差
mean_responses = normalized_replicates.mean(axis=1)
std_errors = normalized_replicates.std(axis=1)

# 對數修正
concentrations = np.where(raw_concentrations <= 0, 1e-6, raw_concentrations)

print(f"🔹 成功讀取到 {len(concentrations)} 組有效濃度數據")
print(f"🔹 自動偵測到重複實驗次數: {replicate_count} 重複")
print("========================================\n")

# 防錯機制
std_errors = np.nan_to_num(std_errors, nan=1e-5)
std_errors = np.where(std_errors == 0, 1e-5, std_errors)

# 4. 進行曲線擬合 (Curve Fitting)
initial_guess = [min(mean_responses), max(mean_responses), np.median(concentrations), 1.0]
bounds = ([0, 0, concentrations.min()*0.1, 0.1], [50, 250, concentrations.max()*10, 5])

try:
    popt, pcov = curve_fit(log_4pl, concentrations, mean_responses, p0=initial_guess, bounds=bounds, sigma=std_errors)
    fitted_min, fitted_max, fitted_ic50, fitted_slope = popt
    ic50_error = np.sqrt(np.diag(pcov))[2]
    
    print("====== 📈 IC50 擬合結果 ======")
    print(f"✅ Bottom (最低相對反應) : {fitted_min:.2f}%")
    print(f"✅ Top (最高相對反應)    : {fitted_max:.2f}%")
    print(f"✅ Hill Slope (斜率)     : {fitted_slope:.2f}")
    print(f"💡 IC50 推估值            : {fitted_ic50:.4f} ± {ic50_error:.4f}")
    print("==============================\n")
    
except Exception as e:
    print("❌ 擬合失敗。原因可能是：數據不符合 S 型趨勢，或者數據波動過大。")
    print(f"詳細錯誤訊息: {e}")
    exit()

# 5. 繪圖與自動存檔
plt.figure(figsize=(8, 6))

# A. 畫出歸一化後的數據點與 Error Bar (黑色外框、白色中心)
plt.errorbar(concentrations, mean_responses, yerr=std_errors, fmt='o', color='black', markeredgecolor='black', markerfacecolor='white', markersize=7, capsize=5, label=f'Normalized Data (Mean ± SD, n={replicate_count})')

# B. 畫出歸一化後的個別原始數據點 (維持半透明淺灰色)
for i in range(replicate_count):
    plt.scatter(concentrations, normalized_replicates[:, i], color='lightgray', alpha=0.5, s=20, zorder=2)

# C. 產生平滑 of X 軸數據來繪製 4PL 擬合曲線
x_smooth = np.logspace(np.log10(concentrations.min()), np.log10(concentrations.max()), 500)
y_smooth = log_4pl(x_smooth, *popt)

# 畫出黑色擬合曲線
plt.plot(x_smooth, y_smooth, '-', color='black', linewidth=2, label='4PL Fitted Curve', zorder=1)

# D. 標註 IC50 位置 (輔助虛線為深灰色)
plt.axvline(x=fitted_ic50, color='dimgray', linestyle='--', alpha=0.6)
plt.axhline(y=(fitted_max + fitted_min)/2, color='dimgray', linestyle='--', alpha=0.6)

# 標註純黑色文字
plt.text(fitted_ic50 * 1.3, (fitted_max + fitted_min)/2 + (fitted_max - fitted_min)*0.05, f'IC50 = {fitted_ic50:.4f}', color='black', fontsize=12, weight='bold')

# E. 調整軸標籤
plt.xscale('log')
plt.xlabel('Concentration (Log Scale)', fontsize=12)
plt.ylabel('Cell Viability (% of Control)', fontsize=12)
plt.title('Dose-Response Curve (Normalized to Control)', fontsize=14, weight='bold')
plt.legend(loc='best')
plt.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()

# 自動儲存圖片
output_image_path = r'c:/Users/PC01/Desktop/ic50 tool/ic50_curve.png'
plt.savefig(output_image_path, dpi=300)
print(f"💾 圖片已更新並儲存至: {output_image_path}")

plt.close()
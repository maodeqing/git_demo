import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import warnings
import duckdb as dd
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

items = ['P1174-7', 'P11774-8', 'P11775-7','P11775-8', 'P11776-7',
        'P11776-8', 'P12092-4', 'P12092-5', 'P12093-5','P12093-4', 
        'P12094-4', 'P12094-5', 'P12297-5','P12297-4']
items = ['P12573-3L','P12573-3H']
df_items = pd.DataFrame(items, columns=['item'])
wb_path = 'C:\\Users\\Dell\\Desktop\\git_demo\\车缝组_多模型预测对比报告(排除周日)\\00_多模型预测综合报告.xlsx'
sht_name = '所有模型預測匯總'
df_forcast = pd.read_excel(wb_path, sheet_name = sht_name)
wb_path = 'C:\\Users\\Dell\\Downloads\\生产每日交货清单 - 複製.xlsx'
sht_name = '生产每日交货清单'
df_production_qty = pd.read_excel(wb_path, sheet_name = sht_name,skipfooter=1)
df_production_qty["开单日期"] = pd.to_datetime(df_production_qty["开单日期"], format='%Y-%m-%d')
df_production_qty = df_production_qty.loc[:,["开单日期", "款号", "发出部门", "交货数"]]
sql = f"SELECT 开单日期, SUM(交货数) as 交货数 from df_production_qty where 发出部门 like '%車縫%' and 款号 in (select item from df_items) group by 开单日期"
df_p12857 = dd.sql(sql).df()
sql = "select forcast.*, pro.交货数 as 觀測數據 from df_forcast as forcast left join df_p12857 as pro on forcast.預測日期=pro.开单日期"
res = dd.execute(sql).df()

res.to_excel("觀測報告.xlsx", index=False)
# 讀取數據
df = pd.read_excel('觀測報告.xlsx', sheet_name='Sheet1')

print("=" * 100)
print("📊 車縫組 P12857 - 預測模型效能評估分析報告")
print("=" * 100)

# 數據清理：只保留有觀測數據的記錄
df_valid = df[df['觀測數據'].notna() & (df['觀測數據'] > 0)].copy()
print(f"\n📋 有效觀測數據筆數: {len(df_valid)} 筆")
print(f"   - 日期範圍: {df_valid['預測日期'].min()} ~ {df_valid['預測日期'].max()}")

# 定義模型列表（排除預測日期、星期、觀測數據列）
model_columns = [col for col in df.columns if col not in ['預測日期', '星期', '觀測數據']]

# ============================================
# 1. 計算各模型的誤差指標
# ============================================
print("\n" + "=" * 100)
print("📈 模型誤差指標計算")
print("=" * 100)

results = []
for model in model_columns:
    # 只使用有觀測數據的行
    y_true = df_valid['觀測數據'].values
    y_pred = df_valid[model].values
    
    # 計算誤差指標
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 計算R平方
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 平均預測值
    avg_pred = np.mean(y_pred)
    avg_actual = np.mean(y_true)
    bias = avg_pred - avg_actual
    
    results.append({
        '模型名稱': model,
        'MAE(件)': round(mae, 0),
        'RMSE(件)': round(rmse, 0),
        'MAPE(%)': round(mape, 1),
        'R²': round(r2, 3),
        '偏誤(Bias)': round(bias, 0),
        '平均預測': round(avg_pred, 0),
        '實際平均': round(avg_actual, 0)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('MAE(件)')

print("\n🏆 模型效能排名（按MAE由低到高）:")
print(results_df[['模型名稱', 'MAE(件)', 'RMSE(件)', 'MAPE(%)', 'R²', '偏誤(Bias)']].to_string(index=False))

# ============================================
# 2. 找出最佳模型
# ============================================
best_model = results_df.iloc[0]['模型名稱']
best_mae = results_df.iloc[0]['MAE(件)']
best_mape = results_df.iloc[0]['MAPE(%)']
best_r2 = results_df.iloc[0]['R²']

print("\n" + "=" * 100)
print(f"🏆 最佳預測模型: {best_model}")
print("=" * 100)
print(f"   ✅ MAE (平均絕對誤差): {best_mae:.0f} 件")
print(f"   ✅ RMSE (均方根誤差): {results_df.iloc[0]['RMSE(件)']:.0f} 件")
print(f"   ✅ MAPE (平均絕對百分比誤差): {best_mape:.1f}%")
print(f"   ✅ R² (決定係數): {best_r2:.3f}")
print(f"   📊 偏誤: {results_df.iloc[0]['偏誤(Bias)']:.0f} 件")

# ============================================
# 3. 各模型類型分析
# ============================================
print("\n" + "=" * 100)
print("📊 各類型模型效能對比")
print("=" * 100)

# 分類模型類型
model_categories = {
    '移動平均': ['SMA-3', 'SMA-7', 'SMA-15', 'SMA-30'],
    '加權移動平均': ['WMA-7', 'WMA-15'],
    '指數平滑': ['ES-α=0.3', 'ES-α=0.5', 'ES-α=0.7'],
    '雙指數平滑': ['Holt-α=0.5,β=0.3', 'Holt-α=0.6,β=0.2'],
    '季節性預測': ['Seasonal-周期7天', 'Seasonal-周期15天', 'Seasonal-周期30天']
}

category_results = []
for category, models in model_categories.items():
    category_models = results_df[results_df['模型名稱'].isin(models)]
    if len(category_models) > 0:
        category_results.append({
            '模型類型': category,
            '平均MAE': round(category_models['MAE(件)'].mean(), 0),
            '平均MAPE(%)': round(category_models['MAPE(%)'].mean(), 1),
            '最佳模型': category_models.iloc[category_models['MAE(件)'].argmin()]['模型名稱'],
            '最佳MAE': category_models['MAE(件)'].min()
        })

category_df = pd.DataFrame(category_results)
category_df = category_df.sort_values('平均MAE')
print(category_df.to_string(index=False))

# ============================================
# 4. 詳細預測vs實際對比圖表
# ============================================
print("\n" + "=" * 100)
print("📈 生成預測 vs 實際對比圖表")
print("=" * 100)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'車縫組 P12857 - 預測模型效能評估 (最佳模型: {best_model})', fontsize=14, fontweight='bold')

# 圖1: 最佳模型預測 vs 實際值
ax = axes[0, 0]
df_sorted = df_valid.sort_values('預測日期')
ax.plot(df_sorted['預測日期'], df_sorted['觀測數據'], 'b-o', linewidth=2, markersize=6, label='實際觀測值')
ax.plot(df_sorted['預測日期'], df_sorted[best_model], 'r--s', linewidth=2, markersize=6, label=f'{best_model} 預測值')
ax.set_title(f'最佳模型: {best_model}\nMAE={best_mae:.0f}, MAPE={best_mape:.1f}%, R²={best_r2:.3f}', fontsize=11, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('交貨數 (件)')
ax.legend()
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# 圖2: 誤差柱狀圖（所有模型）
ax = axes[0, 1]
top_models = results_df.head(8)
colors = ['green' if i == 0 else 'steelblue' for i in range(len(top_models))]
bars = ax.barh(top_models['模型名稱'], top_models['MAE(件)'], color=colors, edgecolor='black')
ax.set_title('模型誤差對比 (MAE - 越低越好)', fontsize=11, fontweight='bold')
ax.set_xlabel('平均絕對誤差 (件)')
ax.axvline(x=top_models['MAE(件)'].mean(), color='red', linestyle='--', alpha=0.7, label=f'平均MAE: {top_models["MAE(件)"].mean():.0f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, top_models['MAE(件)']):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, f'{val:.0f}', va='center', fontsize=9)

# 圖3: 預測誤差分布（最佳模型）
ax = axes[1, 0]
errors = df_valid['觀測數據'].values - df_valid[best_model].values
ax.hist(errors, bins=15, color='skyblue', edgecolor='navy', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零誤差線')
ax.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'平均誤差: {np.mean(errors):.0f}')
ax.set_title(f'{best_model} 預測誤差分布', fontsize=11, fontweight='bold')
ax.set_xlabel('預測誤差 (件)')
ax.set_ylabel('頻次')
ax.legend()
ax.grid(True, alpha=0.3)

# 圖4: 散點圖 - 預測 vs 實際
ax = axes[1, 1]
ax.scatter(df_valid['觀測數據'], df_valid[best_model], alpha=0.7, s=80, c='steelblue', edgecolors='navy')
# 添加完美預測線
max_val = max(df_valid['觀測數據'].max(), df_valid[best_model].max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美預測線')
ax.set_title(f'{best_model} - 預測值 vs 實際值', fontsize=11, fontweight='bold')
ax.set_xlabel('實際交貨數 (件)')
ax.set_ylabel('預測交貨數 (件)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('模型效能評估對比圖.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   ✅ 效能評估圖表已儲存: 模型效能評估對比圖.png")

# ============================================
# 5. 生成各模型詳細預測對比圖
# ============================================
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('車縫組 P12857 - 所有模型預測 vs 實際對比', fontsize=14, fontweight='bold')

for idx, model in enumerate(model_columns[:6]):  # 顯示前6個模型
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    df_sorted = df_valid.sort_values('預測日期')
    ax.plot(df_sorted['預測日期'], df_sorted['觀測數據'], 'b-o', linewidth=2, markersize=5, label='實際觀測值')
    ax.plot(df_sorted['預測日期'], df_sorted[model], 'r--s', linewidth=2, markersize=5, label=f'{model}')
    
    # 計算該模型的MAE
    model_mae = results_df[results_df['模型名稱'] == model]['MAE(件)'].values[0]
    ax.set_title(f'{model} (MAE: {model_mae:.0f}件)', fontsize=10, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('交貨數 (件)')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('所有模型預測對比圖.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   ✅ 所有模型對比圖已儲存: 所有模型預測對比圖.png")

# ============================================
# 6. 生成模型推薦建議
# ============================================
print("\n" + "=" * 100)
print("💡 模型推薦建議")
print("=" * 100)

# 找出第二、第三好的模型
second_best = results_df.iloc[1]['模型名稱'] if len(results_df) > 1 else None
third_best = results_df.iloc[2]['模型名稱'] if len(results_df) > 2 else None

print(f"""
📌 最佳模型: {best_model}
   - 優點: 預測誤差最小，準確度最高
   - 適用場景: 一般生產規劃與預測
   - MAE: {best_mae:.0f}件 | MAPE: {best_mape:.1f}%

📌 次佳模型: {second_best}
   - 可作為備用方案或模型集成使用

📌 各類型模型推薦:
""")

for _, row in category_df.iterrows():
    print(f"   • {row['模型類型']}: {row['最佳模型']} (MAE: {row['最佳MAE']:.0f}件)")

print(f"""
📊 模型選擇建議:
   1. 優先使用 {best_model} 進行預測
   2. 若需要預測穩定性，可考慮 {results_df[results_df['模型名稱'].str.contains('SMA')].iloc[0]['模型名稱'] if len(results_df[results_df['模型名稱'].str.contains('SMA')]) > 0 else 'SMA模型'}
   3. 若數據有季節性趨勢，可參考 {category_df[category_df['模型類型'] == '季節性預測']['最佳模型'].values[0] if len(category_df[category_df['模型類型'] == '季節性預測']) > 0 else '季節性模型'}
   4. 建議使用多模型集成（取平均）來降低單一模型風險
""")

# ============================================
# 7. 生成Excel詳細報告
# ============================================
output_file = '模型效能評估報告.xlsx'
try:
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 模型排名表
        results_df.to_excel(writer, sheet_name='模型排名', index=False)
        
        # 各模型類型匯總
        category_df.to_excel(writer, sheet_name='模型類型匯總', index=False)
        
        # 預測vs實際明細
        detail_df = df_valid[['預測日期', '星期', '觀測數據'] + model_columns].copy()
        detail_df.to_excel(writer, sheet_name='預測vs實際明細', index=False)
        
        # 每日誤差分析
        error_analysis = df_valid[['預測日期', '星期', '觀測數據']].copy()
        for model in model_columns:
            error_analysis[f'{model}_誤差'] = df_valid['觀測數據'] - df_valid[model]
            error_analysis[f'{model}_絕對誤差'] = np.abs(df_valid['觀測數據'] - df_valid[model])
        error_analysis.to_excel(writer, sheet_name='每日誤差分析', index=False)
        
        # 模型推薦
        recommendation = pd.DataFrame({
            '推薦類型': ['最佳準確度', '備用方案', '最穩定(移動平均)', '季節性考量', '保守預測', '樂觀預測'],
            '推薦模型': [
                best_model,
                second_best if second_best else 'N/A',
                results_df[results_df['模型名稱'].str.contains('SMA')].iloc[0]['模型名稱'] if len(results_df[results_df['模型名稱'].str.contains('SMA')]) > 0 else 'SMA-7',
                category_df[category_df['模型類型'] == '季節性預測']['最佳模型'].values[0] if len(category_df[category_df['模型類型'] == '季節性預測']) > 0 else 'N/A',
                results_df[results_df['平均預測'] == results_df['平均預測'].min()]['模型名稱'].values[0] if len(results_df) > 0 else 'N/A',
                results_df[results_df['平均預測'] == results_df['平均預測'].max()]['模型名稱'].values[0] if len(results_df) > 0 else 'N/A'
            ],
            'MAE(件)': [
                best_mae,
                results_df.iloc[1]['MAE(件)'] if len(results_df) > 1 else 'N/A',
                results_df[results_df['模型名稱'].str.contains('SMA')].iloc[0]['MAE(件)'] if len(results_df[results_df['模型名稱'].str.contains('SMA')]) > 0 else 'N/A',
                category_df[category_df['模型類型'] == '季節性預測']['最佳MAE'].values[0] if len(category_df[category_df['模型類型'] == '季節性預測']) > 0 else 'N/A',
                results_df[results_df['平均預測'] == results_df['平均預測'].min()]['MAE(件)'].values[0] if len(results_df) > 0 else 'N/A',
                results_df[results_df['平均預測'] == results_df['平均預測'].max()]['MAE(件)'].values[0] if len(results_df) > 0 else 'N/A'
            ]
        })
        recommendation.to_excel(writer, sheet_name='模型推薦', index=False)
    
    print(f"\n✅ Excel詳細報告已儲存: {output_file}")
except Exception as e:
    print(f"\n⚠️ Excel儲存失敗: {e}")

# ============================================
# 8. 最終摘要
# ============================================
print("\n" + "=" * 100)
print("📋 最終評估摘要")
print("=" * 100)

print(f"""
┌────────────────────────────────────────────────────────────────────┐
│                        模型效能評估結果                              │
├────────────────────────────────────────────────────────────────────┤
│  🏆 最佳模型: {best_model:<35} │
│  📊 MAE: {best_mae:.0f} 件                                                    │
│  📊 MAPE: {best_mape:.1f}%                                                   │
│  📊 R²: {best_r2:.3f}                                                        │
├────────────────────────────────────────────────────────────────────┤
│  📌 第二名: {second_best if second_best else 'N/A':<35} │
│  📌 第三名: {third_best if third_best else 'N/A':<35} │
├────────────────────────────────────────────────────────────────────┤
│  💡 建議:                                                           │
│     1. 優先使用 {best_model} 進行日常預測                            │
│     2. 預測誤差約 {best_mae:.0f} 件，可接受範圍內                      │
│     3. MAPE為 {best_mape:.1f}%，預測準確度良好                         │
└────────────────────────────────────────────────────────────────────┘
""")

print("\n✨ 模型效能評估完成！")
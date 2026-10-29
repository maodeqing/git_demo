import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出資料夾
output_folder = '车缝组_P12857_移动平均预测报告'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"📁 創建資料夾: {output_folder}")

# ============================================
# 1. 讀取 Excel 資料
# ============================================
print("=" * 80)
print("📊 車縫組 - 款號 P12857* 移動平均預測報告")
print("=" * 80)

# 讀取 Excel 檔案
file_path = r'C:\Users\Dell\Downloads\生产每日交货清单.xlsx'
print(f"\n📂 讀取檔案: {file_path}")

try:
    df_raw = pd.read_excel(file_path, sheet_name='生产每日交货清单')
    print(f"✅ 成功讀取 {len(df_raw)} 筆資料")
except FileNotFoundError:
    print(f"❌ 檔案不存在: {file_path}")
    print("請確認檔案路徑是否正確")
    exit()
except Exception as e:
    print(f"❌ 讀取失敗: {e}")
    exit()

# 檢查必要欄位
required_columns = ['开单日期', '发出部门', '交货数']
for col in required_columns:
    if col not in df_raw.columns:
        print(f"❌ 缺少必要欄位: {col}")
        exit()

# 資料清理：只保留交貨數 > 0 的記錄（交貨為0的不納入計算）
df_raw['开单日期'] = pd.to_datetime(df_raw['开单日期'])
df_raw = df_raw.dropna(subset=['开单日期', '发出部门', '交货数'])
df_raw = df_raw[df_raw['交货数'] > 0]  # 關鍵：只保留交貨數 > 0 的記錄
df_raw = df_raw.sort_values('开单日期')

# ============================================
# 2. 篩選條件: 發出部門 LIKE '%車縫%' AND 款號 LIKE 'P12857%'
# ============================================
print("\n" + "=" * 80)
print("🔍 篩選條件:")
print("   - 發出部門 包含「車縫」")
print("   - 款號 以「P12857」開頭")
print("   - 交貨數 > 0（只計算有交貨的日期）")
print("=" * 80)

# 篩選發出部門包含「車縫」的資料
sewing_mask = df_raw['发出部门'].str.contains('車縫', na=False)
df_sewing = df_raw[sewing_mask].copy()

print(f"\n📋 步驟1 - 篩選發出部門包含「車縫」:")
print(f"   - 符合條件的部門: {df_sewing['发出部门'].unique().tolist()}")
print(f"   - 有交貨的資料筆數: {len(df_sewing)} 筆")

# 篩選款號以 P12857 開頭的資料
if '款号' in df_sewing.columns:
    p12857_mask = df_sewing['款号'].astype(str).str.startswith('P12857', na=False)
    df_filtered = df_sewing[p12857_mask].copy()
    print(f"\n📋 步驟2 - 篩選款號以「P12857」開頭:")
    print(f"   - 符合條件的款號: {df_filtered['款号'].unique().tolist()}")
    print(f"   - 有交貨的資料筆數: {len(df_filtered)} 筆")
else:
    print(f"\n⚠️ 資料中沒有「款號」欄位，將僅以發出部門包含「車縫」進行分析")
    df_filtered = df_sewing.copy()

# 檢查是否有資料
if len(df_filtered) == 0:
    print(f"\n❌ 沒有符合條件的資料！")
    exit()

print(f"\n✅ 最終篩選結果:")
print(f"   - 有交貨記錄筆數: {len(df_filtered)} 筆")
print(f"   - 日期範圍: {df_filtered['开单日期'].min().strftime('%Y-%m-%d')} ~ {df_filtered['开单日期'].max().strftime('%Y-%m-%d')}")
print(f"   - 總交貨數: {df_filtered['交货数'].sum():,} 件")
print(f"   - 涉及部門: {df_filtered['发出部门'].unique().tolist()}")
if '款号' in df_filtered.columns:
    print(f"   - 涉及款號: {df_filtered['款号'].unique().tolist()}")

# ============================================
# 3. 準備每日彙總資料（只計算有交貨的天數，不補0）
# ============================================
print("\n" + "=" * 80)
print("📊 準備每日彙總資料（只計算有交貨的天數）")
print("=" * 80)

# 按日期彙總每日交貨量（只彙總有交貨的日期，不補0）
daily_total = df_filtered.groupby('开单日期')['交货数'].sum()

# 重要：這裡只計算有交貨的天數，不補0
# 總天數 = 有交貨的天數
total_days = len(daily_total)
active_days = total_days  # 因為只保留有交貨的日期，所以活躍天數 = 總天數

print(f"\n📊 天數統計（只計算有交貨的日期）:")
print(f"   - 有交貨的天數（總天數）: {total_days} 天")
print(f"   - 活躍天數: {active_days} 天")
print(f"   - 交貨為0的天數: 不納入計算")
print(f"   - 日期範圍: {daily_total.index.min().strftime('%Y-%m-%d')} ~ {daily_total.index.max().strftime('%Y-%m-%d')}")

# 計算產能統計（只基於有交貨的天數）
total_delivery = daily_total.sum()
daily_avg = daily_total.mean()

print(f"\n📊 產能統計:")
print(f"   - 總交貨量: {total_delivery:,.0f} 件")
print(f"   - 日均產能（基於有交貨天數）: {daily_avg:.0f} 件/天")
print(f"   - 最大單日交貨: {daily_total.max():,.0f} 件")
print(f"   - 最小單日交貨: {daily_total.min():,.0f} 件")
print(f"   - 標準差: {daily_total.std():.0f} 件")

# ============================================
# 4. 定義移動平均預測函數（基於有交貨的天數）
# ============================================
def moving_average_forecast(data, window, forecast_days=30):
    """使用移動平均進行預測（基於有交貨的天數）"""
    
    # 計算移動平均（只計算有交貨的數據點）
    ma = data.rolling(window=window, min_periods=1).mean()
    
    # 最後一個移動平均值作為預測基準
    last_ma = ma.iloc[-1]
    
    # 預測未來（使用最後的移動平均值）
    forecast = [last_ma] * forecast_days
    
    return {
        'window': window,
        'forecast': forecast,
        'last_ma': last_ma,
        'ma_series': ma,
        '历史日均': data.mean(),
        '历史总产能': data.sum(),
        '历史天数': len(data)
    }

# ============================================
# 5. 計算不同窗口的移動平均預測
# ============================================
print("\n" + "=" * 80)
print("🔮 移動平均預測 (3天、7天、15天、30天)")
print("=" * 80)

windows = [3, 7, 15, 30]
forecast_days = 30
forecast_results = {}

for window in windows:
    result = moving_average_forecast(daily_total, window, forecast_days)
    forecast_results[window] = result
    
    # 計算預測總產能
    pred_total = sum(result['forecast'])
    pred_daily_avg = pred_total / forecast_days
    
    # 計算變化率
    change_pct = ((pred_daily_avg - result['历史日均']) / result['历史日均'] * 100) if result['历史日均'] > 0 else 0
    
    print(f"\n📌 {window}天移動平均:")
    print(f"   - 歷史有交貨天數: {result['历史天数']} 天")
    print(f"   - 最後移動平均值: {result['last_ma']:.0f} 件/天")
    print(f"   - 歷史日均產能: {result['历史日均']:.0f} 件/天")
    print(f"   - 預測30天總產能: {pred_total:.0f} 件")
    print(f"   - 預測30天日均產能: {pred_daily_avg:.0f} 件/天")
    print(f"   - 變化率: {change_pct:+.1f}%")

# ============================================
# 6. 分別生成3、7、15、30天移動平均報告
# ============================================
print("\n" + "=" * 80)
print("📁 分別生成移動平均預測報告")
print("=" * 80)

for window in windows:
    result = forecast_results[window]
    
    print(f"\n{'='*60}")
    print(f"📌 正在生成 {window}天移動平均預測報告")
    print(f"{'='*60}")
    
    # 計算預測統計
    pred_total = sum(result['forecast'])
    pred_daily_avg = pred_total / forecast_days
    change_pct = ((pred_daily_avg - result['历史日均']) / result['历史日均'] * 100) if result['历史日均'] > 0 else 0
    
    # 準備預測日期（從最後一個有交貨的日期後一天開始）
    future_dates = pd.date_range(start=daily_total.index.max() + timedelta(days=1), periods=forecast_days, freq='D')
    
    # 每日預測明細
    forecast_table = pd.DataFrame({
        '預測日期': future_dates,
        '預測交貨數': [round(v) for v in result['forecast']]
    })
    
    # 移動平均歷史數據
    ma_history = pd.DataFrame({
        '日期': daily_total.index,
        '實際交貨數': daily_total.values,
        f'{window}天移動平均': result['ma_series'].values
    })
    
    # 生成檔名
    safe_name = f'车缝组_P12857_{window}天MA'
    excel_file = os.path.join(output_folder, f'{safe_name}_预测报告.xlsx')
    
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 預測摘要
            summary_df = pd.DataFrame([{
                '預測模型': f'{window}天移動平均',
                '移動平均窗口': f'{window}天',
                '歷史有交貨天數': result['历史天数'],
                '歷史總產能(件)': result['历史总产能'],
                '歷史日均產能(件/天)': round(result['历史日均'], 0),
                '最後移動平均值(件/天)': round(result['last_ma'], 0),
                '預測30天總產能(件)': round(pred_total, 0),
                '預測30天日均產能(件/天)': round(pred_daily_avg, 0),
                '產能變化率(%)': round(change_pct, 1),
                '預測起始日': future_dates[0].strftime('%Y-%m-%d'),
                '預測結束日': future_dates[-1].strftime('%Y-%m-%d'),
                '備註': '只計算有交貨的日期，交貨為0的天數不納入計算'
            }])
            summary_df.to_excel(writer, sheet_name='預測摘要', index=False)
            
            # 每日預測明細
            forecast_table.to_excel(writer, sheet_name='每日預測明細', index=False)
            
            # 移動平均歷史數據
            ma_history.to_excel(writer, sheet_name='移動平均歷史數據', index=False)
            
            # 原始篩選資料
            df_filtered.to_excel(writer, sheet_name='原始資料', index=False)
            
            # 統計分析
            stats_df = pd.DataFrame({
                '統計項目': [
                    '總交貨件數', 
                    '總交易次數', 
                    '平均每次交貨', 
                    '最大單次交貨', 
                    '最小單次交貨', 
                    '標準差',
                    '有交貨天數',
                    '歷史日均產能(基於有交貨天數)'
                ],
                '數值': [
                    f"{df_filtered['交货数'].sum():,}",
                    len(df_filtered),
                    f"{df_filtered['交货数'].mean():.0f}",
                    f"{df_filtered['交货数'].max():,}",
                    f"{df_filtered['交货数'].min():,}",
                    f"{df_filtered['交货数'].std():.0f}",
                    result['历史天数'],
                    f"{result['历史日均']:.0f}"
                ]
            })
            stats_df.to_excel(writer, sheet_name='統計分析', index=False)
            
            # 款號統計（如果有）
            if '款号' in df_filtered.columns:
                style_stats = df_filtered.groupby(['款号'])['交货数'].agg(['sum', 'mean', 'count']).round(2)
                style_stats.columns = ['總交貨數', '平均交貨數', '交易次數']
                style_stats = style_stats.sort_values('總交貨數', ascending=False)
                style_stats.to_excel(writer, sheet_name='款號統計')
            
            # 部門統計
            dept_stats = df_filtered.groupby(['发出部门'])['交货数'].agg(['sum', 'mean', 'count']).round(2)
            dept_stats.columns = ['總交貨數', '平均交貨數', '交易次數']
            dept_stats.to_excel(writer, sheet_name='部門統計')
        
        print(f"   ✅ Excel 報告已儲存: {excel_file}")
        
    except Exception as e:
        print(f"   ⚠️ Excel 匯出失敗: {e}")
    
    # ============================================
    # 7. 繪製該窗口的預測圖表
    # ============================================
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 圖1: 歷史數據 + 移動平均
        axes[0,0].plot(daily_total.index, daily_total.values, 'b-o', linewidth=1, markersize=3, alpha=0.7, label='實際交貨量')
        axes[0,0].plot(daily_total.index, result['ma_series'], 'r-', linewidth=2, label=f'{window}天移動平均')
        axes[0,0].set_title(f'歷史數據與{window}天移動平均\n(共{len(daily_total)}個有交貨日期)', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('日期')
        axes[0,0].set_ylabel('交貨數 (件)')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 圖2: 每月交貨趨勢（加上數量標籤）
        df_filtered['年月'] = df_filtered['开单日期'].dt.strftime('%Y-%m')
        monthly_total = df_filtered.groupby('年月')['交货数'].sum()
        months = range(len(monthly_total))
        month_labels = [str(m) for m in monthly_total.index]
        
        bars = axes[0,1].bar(months, monthly_total.values, color='skyblue', edgecolor='navy', linewidth=1)
        axes[0,1].set_title('每月交貨趨勢', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('月份')
        axes[0,1].set_ylabel('交貨數 (件)')
        axes[0,1].set_xticks(months)
        axes[0,1].set_xticklabels(month_labels, rotation=45)
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, monthly_total.values):
            height = bar.get_height()
            if val >= 10000:
                label = f'{val/10000:.1f}萬'
            else:
                label = f'{int(val):,}'
            axes[0,1].text(bar.get_x() + bar.get_width()/2, height + max(monthly_total.values)*0.02,
                          label, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 圖3: 交貨次數分布
        bins = [0, 500, 1000, 2000, 5000, 10000, 50000]
        labels = ['<500', '500-1000', '1000-2000', '2000-5000', '5000-10000', '>10000']
        df_filtered['交貨區間'] = pd.cut(df_filtered['交货数'], bins=bins, labels=labels)
        count_dist = df_filtered['交貨區間'].value_counts().sort_index()
        
        bars = axes[1,0].bar(range(len(count_dist)), count_dist.values, color='lightcoral', edgecolor='darkred')
        axes[1,0].set_title('交貨次數分布', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('交貨數量區間 (件)')
        axes[1,0].set_ylabel('次數')
        axes[1,0].set_xticks(range(len(count_dist)))
        axes[1,0].set_xticklabels(count_dist.index, rotation=45)
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, count_dist.values):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 圖4: 產能預測
        future_dates = pd.date_range(start=daily_total.index.max() + timedelta(days=1), periods=forecast_days, freq='D')
        forecast_values = result['forecast']
        
        # 顯示最近30天歷史數據
        last_30_days = daily_total.tail(30)
        if len(last_30_days) > 0:
            axes[1,1].plot(last_30_days.index, last_30_days.values, 'b-o', linewidth=2, markersize=4, label='歷史數據 (過去30個有交貨日)')
        
        axes[1,1].plot(future_dates, forecast_values, 'r--o', linewidth=2, markersize=4, label=f'{window}天移動平均預測')
        
        # 添加預測值標籤
        for i, (date, val) in enumerate(zip(future_dates, forecast_values)):
            if i % 5 == 0 or i == forecast_days - 1:
                axes[1,1].annotate(f'{int(val):,}', xy=(date, val), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=8, alpha=0.7)
        
        # 添加平均值線
        hist_avg = result['历史日均']
        pred_avg = pred_daily_avg
        axes[1,1].axhline(y=hist_avg, color='blue', linestyle=':', alpha=0.5, label=f'歷史日均: {hist_avg:.0f}件')
        axes[1,1].axhline(y=pred_avg, color='red', linestyle=':', alpha=0.5, label=f'預測日均: {pred_avg:.0f}件')
        
        axes[1,1].set_title(f'{window}天移動平均預測 (未來30天)\n預測30天總產能: {pred_total:,.0f}件 | 變化率: {change_pct:+.1f}%', 
                           fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('日期')
        axes[1,1].set_ylabel('交貨數 (件)')
        axes[1,1].legend(loc='upper left')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        image_file = os.path.join(output_folder, f'{safe_name}_预测图表.png')
        plt.savefig(image_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ 預測圖表已儲存: {image_file}")
        
    except Exception as e:
        print(f"   ⚠️ 圖表繪製失敗: {e}")
        plt.close()

# ============================================
# 8. 生成比較匯總報告
# ============================================
print("\n" + "=" * 80)
print("📊 生成移動平均預測比較匯總報告")
print("=" * 80)

# 匯總所有預測結果
comparison_data = []
for window in windows:
    result = forecast_results[window]
    pred_total = sum(result['forecast'])
    pred_daily = pred_total / forecast_days
    hist_daily = result['历史日均']
    change_pct = ((pred_daily - hist_daily) / hist_daily * 100) if hist_daily > 0 else 0
    
    comparison_data.append({
        '移動平均窗口': f'{window}天',
        '歷史有交貨天數': result['历史天数'],
        '最後移動平均值(件/天)': round(result['last_ma'], 0),
        '歷史日均產能(件/天)': round(hist_daily, 0),
        '預測30天總產能(件)': round(pred_total, 0),
        '預測30天日均產能(件/天)': round(pred_daily, 0),
        '產能變化率(%)': round(change_pct, 1),
        '預測穩定性': '高' if window >= 15 else '中' if window >= 7 else '低'
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_file = os.path.join(output_folder, '00_移动平均预测比较汇总.xlsx')
try:
    with pd.ExcelWriter(comparison_file, engine='openpyxl') as writer:
        comparison_df.to_excel(writer, sheet_name='預測比較', index=False)
        
        # 添加計算說明
        note_df = pd.DataFrame({
            '說明項目': ['計算方式', '總天數定義', '活躍天數定義', '日均產能計算', '移動平均計算'],
            '說明內容': [
                '只統計交貨數 > 0 的記錄',
                '有交貨記錄的天數（不含交貨為0的天數）',
                '有交貨記錄的天數（與總天數相同）',
                '總交貨量 ÷ 有交貨天數',
                '只基於有交貨的日期進行計算'
            ]
        })
        note_df.to_excel(writer, sheet_name='計算說明', index=False)
        
        # 各窗口建議
        recommendation = pd.DataFrame({
            '移動平均窗口': ['3天', '7天', '15天', '30天'],
            '適用場景': [
                '短期波動大，適合快速反應，但預測穩定性較低',
                '平衡短期波動和穩定性，適合一般生產規劃',
                '平滑效果較好，適合中期生產計劃',
                '最平滑，適合長期趨勢分析，但反應較慢'
            ],
            '建議使用': ['適合日/週計劃', '適合週/月計劃', '適合月/季計劃', '適合季/年計劃']
        })
        recommendation.to_excel(writer, sheet_name='建議說明', index=False)
    
    print(f"✅ 比較匯總報告已儲存: {comparison_file}")
except Exception as e:
    print(f"⚠️ 比較匯總報告儲存失敗: {e}")

# ============================================
# 9. 最終摘要報告
# ============================================
print("\n" + "=" * 80)
print("📋 移動平均預測最終摘要報告")
print("=" * 80)

print(f"\n📊 資料篩選條件:")
print(f"   - 發出部門: 包含「車縫」")
print(f"   - 款號: 以「P12857」開頭")
print(f"   - 交貨數: > 0（只計算有交貨的記錄）")

print(f"\n📊 歷史資料統計（只計算有交貨的日期）:")
print(f"   - 有交貨天數: {total_days} 天")
print(f"   - 資料期間: {daily_total.index.min().strftime('%Y-%m-%d')} ~ {daily_total.index.max().strftime('%Y-%m-%d')}")
print(f"   - 歷史總產能: {total_delivery:,.0f} 件")
print(f"   - 歷史日均產能: {daily_avg:.0f} 件/天")

print(f"\n🔮 各窗口預測結果比較:")
print(f"{'='*80}")
print(f"{'窗口':<8} {'有交貨天數':<12} {'歷史日均':<12} {'預測總產能':<15} {'預測日均':<12} {'變化率':<10}")
print(f"{'-'*80}")

for window in windows:
    result = forecast_results[window]
    pred_total = sum(result['forecast'])
    pred_daily = pred_total / forecast_days
    hist_daily = result['历史日均']
    change_pct = ((pred_daily - hist_daily) / hist_daily * 100) if hist_daily > 0 else 0
    print(f"{window}天{'':<5} {result['历史天数']:>8}    {hist_daily:>8,.0f}    {pred_total:>12,.0f}    {pred_daily:>8,.0f}    {change_pct:>+7.1f}%")

print(f"{'='*80}")

print(f"\n📁 輸出檔案結構:")
print(f"\n{output_folder}/")
print("   ├── 00_移动平均预测比较汇总.xlsx    (所有窗口比較)")
print("   ├── 车缝组_P12857_3天MA_预测报告.xlsx")
print("   ├── 车缝组_P12857_3天MA_预测图表.png")
print("   ├── 车缝组_P12857_7天MA_预测报告.xlsx")
print("   ├── 车缝组_P12857_7天MA_预测图表.png")
print("   ├── 车缝组_P12857_15天MA_预测报告.xlsx")
print("   ├── 车缝组_P12857_15天MA_预测图表.png")
print("   ├── 车缝组_P12857_30天MA_预测报告.xlsx")
print("   └── 车缝组_P12857_30天MA_预测图表.png")

print("\n" + "=" * 80)
print("⚠️ 重要說明:")
print("   1. 本報告只統計交貨數 > 0 的記錄")
print("   2. 交貨為0的天數不納入計算（不影響總天數和活躍天數）")
print("   3. 日均產能 = 總交貨量 ÷ 有交貨天數")
print("   4. 移動平均計算只基於有交貨的日期")
print("   5. 3天MA反應最快，30天MA最平滑")
print("   6. 建議根據實際生產需求選擇合適的預測窗口")
print("=" * 80)

print("\n✨ 所有移動平均預測報告生成完成！")
print(f"📁 報告已儲存至: {os.path.abspath(output_folder)}")
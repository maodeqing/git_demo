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
output_folder = '部门产能报告_移动平均'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"📁 創建資料夾: {output_folder}")

# ============================================
# 1. 讀取 Excel 資料
# ============================================
print("=" * 80)
print("📊 各發出部門 - 移動平均產能預測報告 (3/7/15/30天)")
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
missing_cols = [col for col in required_columns if col not in df_raw.columns]
if missing_cols:
    print(f"❌ 缺少必要欄位: {missing_cols}")
    print(f"檔案中現有欄位: {df_raw.columns.tolist()}")
    exit()

# 資料清理：只保留交貨數 > 0 的記錄
df_raw['开单日期'] = pd.to_datetime(df_raw['开单日期'])
df_raw = df_raw.dropna(subset=['开单日期', '发出部门', '交货数'])
df_raw = df_raw[df_raw['交货数'] > 0]
df_raw = df_raw.sort_values('开单日期')

# 獲取所有部門
departments = df_raw['发出部门'].unique()
print(f"\n📋 發現 {len(departments)} 個發出部門:")
for i, dept in enumerate(departments, 1):
    dept_data = df_raw[df_raw['发出部门'] == dept]
    active_days = dept_data['开单日期'].dt.date.nunique()
    print(f"   {i}. {dept} ({len(dept_data)} 筆資料, {active_days} 活躍天數, 總交貨: {dept_data['交货数'].sum():,} 件)")

# ============================================
# 2. 定義移動平均預測函數
# ============================================
def moving_average_forecast(daily_values, window, forecast_days=30):
    """使用移動平均進行預測（基於有交貨的數據點）"""
    
    if len(daily_values) < window:
        window = len(daily_values)
    
    # 計算移動平均
    ma_series = pd.Series(daily_values).rolling(window=window, min_periods=1).mean()
    
    # 最後一個移動平均值作為預測基準
    last_ma = ma_series.iloc[-1]
    
    # 計算歷史日均
    hist_avg = np.mean(daily_values)
    hist_total = np.sum(daily_values)
    
    # 預測未來
    forecast = [last_ma] * forecast_days
    
    # 添加小幅度波動
    np.random.seed(42)
    noise_level = min(0.1, np.std(daily_values) / (last_ma + 1) * 0.3) if len(daily_values) > 5 else 0.05
    noise = np.random.normal(0, last_ma * noise_level, forecast_days)
    forecast = [max(0, f + n) for f, n in zip(forecast, noise)]
    
    return {
        'window': window,
        'forecast': forecast,
        'last_ma': last_ma,
        'ma_series': ma_series,
        '历史日均': hist_avg,
        '历史总产能': hist_total,
        '历史天数': len(daily_values),
        '预测总产能': np.sum(forecast),
        '预测日均': np.mean(forecast)
    }


# ============================================
# 3. 對每個部門進行預測
# ============================================
print("\n" + "=" * 80)
print("🔮 開始為每個部門生成移動平均預測報告...")
print("=" * 80)

windows = [3, 7, 15, 30]
forecast_days = 30
all_results = {}

for dept in departments:
    print(f"\n{'='*60}")
    print(f"📌 正在處理: {dept}")
    print(f"{'='*60}")
    
    df_dept = df_raw[df_raw['发出部门'] == dept].copy()
    
    # 按日期彙總每日交貨量（只彙總有交貨的日期）
    daily_total = df_dept.groupby('开单日期')['交货数'].sum()
    daily_values = daily_total.values
    active_days = len(daily_total)
    
    print(f"\n   📊 {dept} 基本統計:")
    print(f"      - 有交貨記錄筆數: {len(df_dept)} 筆")
    print(f"      - 有交貨天數（活躍天數）: {active_days} 天")
    print(f"      - 總交貨量: {daily_total.sum():,.0f} 件")
    print(f"      - 日均產能（基於有交貨天數）: {daily_total.mean():.0f} 件/天")
    
    # 計算各窗口的移動平均預測
    dept_results = {}
    for window in windows:
        result = moving_average_forecast(daily_values, window, forecast_days)
        dept_results[window] = result
        
        print(f"\n   📌 {window}天移動平均:")
        print(f"      - 最後移動平均值: {result['last_ma']:.0f} 件/天")
        print(f"      - 歷史日均: {result['历史日均']:.0f} 件/天")
        print(f"      - 預測30天總產能: {result['预测总产能']:.0f} 件")
        print(f"      - 預測30天日均: {result['预测日均']:.0f} 件/天")
    
    all_results[dept] = {
        'df_dept': df_dept,
        'daily_total': daily_total,
        'daily_values': daily_values,
        'active_days': active_days,
        'forecasts': dept_results
    }
    
    # ============================================
    # 4. 生成該部門的 Excel 報告
    # ============================================
    
    # 準備各窗口預測結果表格
    future_dates = pd.date_range(start=daily_total.index.max() + timedelta(days=1), periods=forecast_days, freq='D')
    
    forecast_comparison = pd.DataFrame({'預測日期': future_dates})
    for window in windows:
        forecast_comparison[f'{window}天MA預測'] = [round(v) for v in all_results[dept]['forecasts'][window]['forecast']]
    
    # 添加平均值行
    avg_row = {'預測日期': '平均值'}
    for window in windows:
        avg_row[f'{window}天MA預測'] = round(all_results[dept]['forecasts'][window]['预测日均'], 0)
    forecast_comparison = pd.concat([forecast_comparison, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 添加時間維度統計
    df_dept['日期'] = df_dept['开单日期'].dt.date
    df_dept['年週'] = df_dept['开单日期'].dt.strftime('%Y-W%W')
    df_dept['年月'] = df_dept['开单日期'].dt.strftime('%Y-%m')
    df_dept['季度'] = df_dept['开单日期'].dt.to_period('Q').astype(str)
    df_dept['年度'] = df_dept['开单日期'].dt.year
    
    daily_stats = df_dept.groupby(['日期'])['交货数'].sum().reset_index().sort_values('日期', ascending=False)
    weekly_stats = df_dept.groupby(['年週'])['交货数'].sum().reset_index().sort_values('年週')
    monthly_stats = df_dept.groupby(['年月'])['交货数'].sum().reset_index().sort_values('年月')
    quarterly_stats = df_dept.groupby(['季度'])['交货数'].sum().reset_index().sort_values('季度')
    yearly_stats = df_dept.groupby(['年度'])['交货数'].sum().reset_index().sort_values('年度')
    
    # 款號統計
    if '款号' in df_dept.columns:
        style_stats = df_dept.groupby(['款号'])['交货数'].agg(['sum', 'mean', 'count']).round(2)
        style_stats.columns = ['總交貨數', '平均交貨數', '交易次數']
        style_stats = style_stats.sort_values('總交貨數', ascending=False)
    else:
        style_stats = None
    
    # 交貨次數分布
    bins = [0, 500, 1000, 2000, 5000, 10000, 50000, 100000]
    labels = ['<500', '500-1000', '1000-2000', '2000-5000', '5000-10000', '10000-50000', '>50000']
    df_dept['交貨區間'] = pd.cut(df_dept['交货数'], bins=bins, labels=labels)
    dist_stats = df_dept['交貨區間'].value_counts().sort_index().reset_index()
    dist_stats.columns = ['交貨區間(件)', '次數']
    
    # 生成檔名
    safe_name = dept.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('組', '组')
    excel_file = os.path.join(output_folder, f'{safe_name}_移动平均预测报告.xlsx')
    
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 預測比較匯總
            summary_data = []
            for window in windows:
                r = all_results[dept]['forecasts'][window]
                change_pct = ((r['预测日均'] - r['历史日均']) / r['历史日均'] * 100) if r['历史日均'] > 0 else 0
                summary_data.append({
                    '移動平均窗口': f'{window}天',
                    '歷史有交貨天數': r['历史天数'],
                    '歷史總產能(件)': round(r['历史总产能'], 0),
                    '歷史日均產能(件/天)': round(r['历史日均'], 0),
                    '最後移動平均值(件/天)': round(r['last_ma'], 0),
                    '預測30天總產能(件)': round(r['预测总产能'], 0),
                    '預測30天日均產能(件/天)': round(r['预测日均'], 0),
                    '產能變化率(%)': round(change_pct, 1)
                })
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='預測比較匯總', index=False)
            
            # 每日預測比較
            forecast_comparison.to_excel(writer, sheet_name='每日預測比較', index=False)
            
            # 原始資料
            df_dept.to_excel(writer, sheet_name='原始資料', index=False)
            
            # 日報
            daily_stats.to_excel(writer, sheet_name='日報', index=False)
            
            # 週報
            weekly_stats.to_excel(writer, sheet_name='週報', index=False)
            
            # 月報
            monthly_stats.to_excel(writer, sheet_name='月報', index=False)
            
            # 季報
            quarterly_stats.to_excel(writer, sheet_name='季報', index=False)
            
            # 年報
            yearly_stats.to_excel(writer, sheet_name='年報', index=False)
            
            # 款號統計
            if style_stats is not None:
                style_stats.to_excel(writer, sheet_name='款號統計')
            
            # 基本統計
            basic_stats = pd.DataFrame({
                '統計項目': [
                    '總交貨件數', '總交易次數', '平均每次交貨', '最大單次交貨', 
                    '最小單次交貨', '標準差', '有交貨天數(活躍天數)', '歷史日均產能(基於有交貨天數)'
                ],
                '數值': [
                    f"{df_dept['交货数'].sum():,}", len(df_dept), f"{df_dept['交货数'].mean():.0f}",
                    f"{df_dept['交货数'].max():,}", f"{df_dept['交货数'].min():,}",
                    f"{df_dept['交货数'].std():.0f}", active_days, f"{daily_total.mean():.0f}"
                ]
            })
            basic_stats.to_excel(writer, sheet_name='基本統計', index=False)
            
            # 交貨次數分布
            dist_stats.to_excel(writer, sheet_name='交貨次數分布', index=False)
            
            # 計算說明
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
        
        print(f"\n   ✅ Excel 報告已儲存: {excel_file}")
        
    except Exception as e:
        print(f"   ⚠️ Excel 匯出失敗: {e}")
    
    # ============================================
    # 5. 繪製該部門的預測比較圖
    # ============================================
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 圖1: 歷史數據 + 各窗口移動平均
        ax1 = axes[0, 0]
        days = range(len(daily_values))
        ax1.plot(days, daily_values, 'k-o', linewidth=1, markersize=3, alpha=0.5, label='實際交貨量')
        
        colors = ['blue', 'green', 'orange', 'red']
        for i, window in enumerate(windows):
            ma_series = all_results[dept]['forecasts'][window]['ma_series']
            ax1.plot(days, ma_series, color=colors[i], linewidth=1.5, label=f'{window}天MA')
        
        ax1.set_title(f'{dept} - 歷史數據與移動平均比較\n(共{active_days}個有交貨日期)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('有交貨日期序號')
        ax1.set_ylabel('交貨數 (件)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 圖2: 每月交貨趨勢
        ax2 = axes[0, 1]
        monthly_total = df_dept.groupby(df_dept['开单日期'].dt.to_period('M'))['交货数'].sum()
        months = range(len(monthly_total))
        month_labels = [str(m) for m in monthly_total.index]
        
        bars = ax2.bar(months, monthly_total.values, color='skyblue', edgecolor='navy')
        ax2.set_title(f'{dept} - 每月交貨趨勢', fontsize=12, fontweight='bold')
        ax2.set_xlabel('月份')
        ax2.set_ylabel('交貨數 (件)')
        ax2.set_xticks(months)
        ax2.set_xticklabels(month_labels, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, monthly_total.values):
            height = bar.get_height()
            label = f'{val/10000:.1f}萬' if val >= 10000 else f'{int(val):,}'
            ax2.text(bar.get_x() + bar.get_width()/2, height + max(monthly_total.values)*0.02,
                    label, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 圖3: 各窗口預測總產能比較
        ax3 = axes[1, 0]
        window_labels = [f'{w}天' for w in windows]
        pred_totals = [all_results[dept]['forecasts'][w]['预测总产能'] for w in windows]
        hist_total = daily_total.sum()
        
        x = np.arange(len(window_labels))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, [hist_total] * len(window_labels), width, label='歷史總產能', color='skyblue')
        bars2 = ax3.bar(x + width/2, pred_totals, width, label='預測30天總產能', color='lightcoral')
        
        ax3.set_title(f'{dept} - 歷史 vs 預測總產能比較', fontsize=12, fontweight='bold')
        ax3.set_xlabel('移動平均窗口')
        ax3.set_ylabel('產能 (件)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(window_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar in bars1:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height()/1000)}k', 
                    ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height()/1000)}k', 
                    ha='center', va='bottom', fontsize=8)
        
        # 圖4: 各窗口預測對比
        ax4 = axes[1, 1]
        
        # 顯示最近30天歷史數據
        last_30 = daily_values[-30:] if len(daily_values) >= 30 else daily_values
        x_hist = range(len(last_30))
        ax4.plot(x_hist, last_30, 'k-o', linewidth=2, markersize=4, label='歷史數據 (最近)')
        
        # 繪製各窗口預測
        for i, window in enumerate(windows):
            forecast_values = all_results[dept]['forecasts'][window]['forecast']
            x_forecast = range(len(last_30), len(last_30) + forecast_days)
            ax4.plot(x_forecast, forecast_values, color=colors[i], linestyle='--', 
                    linewidth=1.5, marker='o', markersize=3, label=f'{window}天MA預測')
        
        ax4.set_title(f'{dept} - 各窗口預測比較 (未來30天)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('有交貨日期序號')
        ax4.set_ylabel('交貨數 (件)')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        image_file = os.path.join(output_folder, f'{safe_name}_移动平均预测图.png')
        plt.savefig(image_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ 預測圖已儲存: {image_file}")
        
    except Exception as e:
        print(f"   ⚠️ 圖表繪製失敗: {e}")
        plt.close()
    
    # 顯示預測結果摘要
    print(f"\n   📊 {dept} 各窗口預測結果比較:")
    print(f"      {'窗口':<8} {'歷史日均':<12} {'預測日均':<12} {'預測總產能':<15} {'變化率':<10}")
    print(f"      {'-'*55}")
    for window in windows:
        r = all_results[dept]['forecasts'][window]
        change_pct = ((r['预测日均'] - r['历史日均']) / r['历史日均'] * 100) if r['历史日均'] > 0 else 0
        print(f"      {window}天{'':<5} {r['历史日均']:>8,.0f}    {r['预测日均']:>8,.0f}    {r['预测总产能']:>12,.0f}    {change_pct:>+7.1f}%")

# ============================================
# 6. 生成所有部門匯總報告
# ============================================
print("\n" + "=" * 80)
print("📊 生成所有部門匯總報告")
print("=" * 80)

# 匯總所有部門的最佳預測（基於30天MA作為代表）
summary_data = []
for dept, data in all_results.items():
    r30 = data['forecasts'][30]
    change_pct = ((r30['预测日均'] - r30['历史日均']) / r30['历史日均'] * 100) if r30['历史日均'] > 0 else 0
    summary_data.append({
        '部門': dept,
        '有交貨天數': data['active_days'],
        '歷史總產能(件)': round(r30['历史总产能'], 0),
        '歷史日均產能(件/天)': round(r30['历史日均'], 0),
        '3天MA預測總產能': round(data['forecasts'][3]['预测总产能'], 0),
        '7天MA預測總產能': round(data['forecasts'][7]['预测总产能'], 0),
        '15天MA預測總產能': round(data['forecasts'][15]['预测总产能'], 0),
        '30天MA預測總產能': round(data['forecasts'][30]['预测总产能'], 0),
        '30天MA變化率(%)': round(change_pct, 1)
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('30天MA預測總產能', ascending=False)

summary_file = os.path.join(output_folder, '00_各部门移动平均预测汇总.xlsx')
try:
    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='各部門預測匯總', index=False)
        
        # 各窗口建議說明
        recommendation = pd.DataFrame({
            '移動平均窗口': ['3天', '7天', '15天', '30天'],
            '特點': [
                '反應最快，波動較大',
                '平衡穩定性和反應速度',
                '平滑效果較好',
                '最平滑，反應較慢'
            ],
            '適用場景': [
                '短期生產計劃，快速反應市場變化',
                '週生產計劃，平衡穩定性',
                '月生產計劃，中期趨勢分析',
                '季/年生產計劃，長期趨勢分析'
            ],
            '建議使用': ['適合日/週計劃', '適合週/月計劃', '適合月/季計劃', '適合季/年計劃']
        })
        recommendation.to_excel(writer, sheet_name='窗口建議說明', index=False)
        
        # 計算說明
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
    
    print(f"✅ 匯總報告已儲存: {summary_file}")
except Exception as e:
    print(f"⚠️ 匯總報告儲存失敗: {e}")

# ============================================
# 7. 最終摘要報告
# ============================================
print("\n" + "=" * 80)
print("📋 各部門移動平均預測最終摘要報告")
print("=" * 80)

total_historical = df_raw['交货数'].sum()
total_active_days = df_raw['开单日期'].dt.date.nunique()
print(f"\n📊 整體統計:")
print(f"   - 部門總數: {len(departments)}")
print(f"   - 成功預測部門數: {len(all_results)}")
print(f"   - 資料總筆數(僅交貨>0): {len(df_raw):,}")
print(f"   - 歷史總交貨量: {total_historical:,.0f} 件")
print(f"   - 整體有交貨天數: {total_active_days} 天")
print(f"   - 整體日均產能: {total_historical/total_active_days:.0f} 件/天")

print(f"\n📌 各窗口特點說明:")
print(f"   {'窗口':<8} {'特點':<20} {'適用場景':<20} {'建議使用'}")
print(f"   {'-'*60}")
print(f"   3天{'':<6} 反應最快，波動較大{'':<6} 短期生產計劃{'':<8} 適合日/週計劃")
print(f"   7天{'':<6} 平衡穩定性和反應速度{'':<4} 週生產計劃{'':<10} 適合週/月計劃")
print(f"   15天{'':<5} 平滑效果較好{'':<12} 月生產計劃{'':<10} 適合月/季計劃")
print(f"   30天{'':<5} 最平滑，反應較慢{'':<8} 季/年生產計劃{'':<8} 適合季/年計劃")

print("\n📌 各部門30天MA預測排名:")
sorted_depts = sorted(all_results.items(), key=lambda x: x[1]['forecasts'][30]['预测总产能'], reverse=True)
for i, (dept, data) in enumerate(sorted_depts, 1):
    r30 = data['forecasts'][30]
    change_pct = ((r30['预测日均'] - r30['历史日均']) / r30['历史日均'] * 100) if r30['历史日均'] > 0 else 0
    growth_symbol = "📈" if change_pct > 3 else "📉" if change_pct < -3 else "➡️"
    print(f"   {i}. {dept}: {r30['预测总产能']:>12,} 件 (日均: {r30['预测日均']:>6,.0f} | 趨勢: {change_pct:>+6.1f}% {growth_symbol})")

print("\n" + "=" * 80)
print("📁 輸出檔案結構:")
print(f"\n{output_folder}/")
print("   ├── 00_各部门移动平均预测汇总.xlsx    (所有部門匯總)")
print("   ├── 車縫2组_移动平均预测报告.xlsx    (車縫2組完整報告)")
print("   ├── 車縫2组_移动平均预测图.png       (車縫2組預測圖)")
print("   ├── 鈕壓_移动平均预测报告.xlsx        (鈕壓完整報告)")
print("   ├── 鈕壓_移动平均预测图.png           (鈕壓預測圖)")
print("   └── ... (其他部門)")
print("=" * 80)

print("\n✨ 所有部門移動平均預測完成！")
print(f"📁 報告已儲存至: {os.path.abspath(output_folder)}")
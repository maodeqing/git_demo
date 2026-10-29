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
output_folder = '部门产能报告'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"📁 創建資料夾: {output_folder}")

# ============================================
# 1. 讀取 Excel 資料
# ============================================
print("=" * 70)
print("📊 各發出部門 - 生產交貨資料分析與產能預測")
print("=" * 70)

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

# 資料清理：只保留交貨數 > 0 的記錄（交貨為0的不納入計算）
df_raw['开单日期'] = pd.to_datetime(df_raw['开单日期'])
df_raw = df_raw.dropna(subset=['开单日期', '发出部门', '交货数'])
df_raw = df_raw[df_raw['交货数'] > 0]  # 關鍵：只保留交貨數 > 0 的記錄
df_raw = df_raw.sort_values('开单日期')

# 獲取所有部門
departments = df_raw['发出部门'].unique()
print(f"\n📋 發現 {len(departments)} 個發出部門:")
for i, dept in enumerate(departments, 1):
    dept_data = df_raw[df_raw['发出部门'] == dept]
    print(f"   {i}. {dept} ({len(dept_data)} 筆資料, 總交貨: {dept_data['交货数'].sum():,} 件)")

# ============================================
# 2. 定義預測函數（只計算有交貨的天數，不補0）
# ============================================
def forecast_dept(df_dept, dept_name, forecast_days=30):
    """對單一部門進行產能預測（只計算有交貨的天數，不補0）"""
    
    if len(df_dept) < 3:
        return None
    
    # 按日期彙總每日交貨量（只彙總有交貨的日期，不補0）
    daily_dept = df_dept.groupby('开单日期')['交货数'].sum()
    
    # 重要：這裡不補0，只保留有交貨的日期
    # daily_dept 的長度 = 有交貨的天數
    
    # 計算歷史統計（只基於有交貨的天數）
    historical_daily_avg = daily_dept.mean()  # 有交貨天數的日均
    historical_total = daily_dept.sum()
    active_days = len(daily_dept)  # 有交貨的天數
    
    # 計算不同時間窗口的移動平均（只基於有交貨的數據點）
    # 將數據轉為數列，窗口基於有交貨的天數
    daily_values = daily_dept.values
    
    # 計算移動平均（基於數據點數，而非日曆天數）
    ma_values = []
    for i in range(len(daily_values)):
        start = max(0, i - 7 + 1)
        ma_7 = np.mean(daily_values[start:i+1])
        ma_values.append(ma_7)
    
    # 最後一個移動平均值作為預測基準
    recent_ma = ma_values[-1] if ma_values else historical_daily_avg
    
    # 計算成長趨勢（基於有交貨的數據點）
    if len(daily_values) >= 28:
        recent_14 = np.mean(daily_values[-14:])
        older_14 = np.mean(daily_values[:14])
        growth_rate = ((recent_14 - older_14) / older_14 * 100) if older_14 > 0 else 0
    elif len(daily_values) >= 14:
        recent_7 = np.mean(daily_values[-7:])
        older_7 = np.mean(daily_values[:7])
        growth_rate = ((recent_7 - older_7) / older_7 * 100) if older_7 > 0 else 0
    else:
        growth_rate = 0
    
    # 預測未來（使用最後的移動平均值）
    forecast = []
    for i in range(forecast_days):
        if abs(growth_rate) > 1:
            trend_factor = 1 + (growth_rate / 100) * (i / forecast_days)
            forecast.append(max(0, recent_ma * trend_factor))
        else:
            forecast.append(max(0, recent_ma))
    
    # 添加自然波動
    if recent_ma > 0 and len(daily_values) > 7:
        np.random.seed(42)
        noise_level = min(0.15, np.std(daily_values) / (recent_ma + 1) * 0.5)
        noise = np.random.normal(0, recent_ma * noise_level, forecast_days)
        forecast = [max(0, f + n) for f, n in zip(forecast, noise)]
    
    # 判斷是否有週期性（星期幾效應）- 需要日期資訊
    has_seasonality = False
    if len(daily_dept) >= 14:
        # 獲取每個有交貨日期的星期幾
        weekdays = [d.dayofweek for d in daily_dept.index]
        if len(set(weekdays)) > 1:
            weekday_avg = {}
            for w in set(weekdays):
                indices = [i for i, day in enumerate(weekdays) if day == w]
                if indices:
                    weekday_avg[w] = np.mean([daily_values[i] for i in indices])
            if weekday_avg:
                avg_values = list(weekday_avg.values())
                has_seasonality = np.std(avg_values) > np.mean(avg_values) * 0.3 if np.mean(avg_values) > 0 else False
    
    return {
        '部门': dept_name,
        '模型': '移動平均預測',
        '历史总产能': round(historical_total, 0),
        '历史日均产能': round(historical_daily_avg, 0),  # 基於有交貨天數
        '活跃天数': active_days,
        '预测日均产能': round(np.mean(forecast), 0),
        '预测总产能': round(np.sum(forecast), 0),
        '增长趋势': round(growth_rate, 1),
        '预测值': forecast,
        '历史数据天数': active_days,
        'has_seasonality': has_seasonality,
        'daily_dept': daily_dept  # 保存每日數據用於繪圖
    }


# ============================================
# 3. 對每個部門進行預測並生成獨立報告
# ============================================
print("\n" + "=" * 70)
print("🔮 開始為每個部門生成獨立報告...")
print("=" * 70)

all_results = []

for dept in departments:
    print(f"\n{'='*50}")
    print(f"📌 正在處理: {dept}")
    print(f"{'='*50}")
    
    df_dept = df_raw[df_raw['发出部门'] == dept].copy()
    
    # 添加時間維度
    df_dept['日期'] = df_dept['开单日期'].dt.date
    df_dept['年週'] = df_dept['开单日期'].dt.strftime('%Y-W%W')
    df_dept['年月'] = df_dept['开单日期'].dt.strftime('%Y-%m')
    df_dept['季度'] = df_dept['开单日期'].dt.to_period('Q').astype(str)
    df_dept['年度'] = df_dept['开单日期'].dt.year
    
    # 計算各時間維度統計
    daily_stats = df_dept.groupby(['日期'])['交货数'].sum().reset_index()
    daily_stats = daily_stats.sort_values('日期', ascending=False)
    
    weekly_stats = df_dept.groupby(['年週'])['交货数'].sum().reset_index()
    weekly_stats = weekly_stats.sort_values('年週')
    
    monthly_stats = df_dept.groupby(['年月'])['交货数'].sum().reset_index()
    monthly_stats = monthly_stats.sort_values('年月')
    
    quarterly_stats = df_dept.groupby(['季度'])['交货数'].sum().reset_index()
    quarterly_stats = quarterly_stats.sort_values('季度')
    
    yearly_stats = df_dept.groupby(['年度'])['交货数'].sum().reset_index()
    yearly_stats = yearly_stats.sort_values('年度')
    
    # 款號統計
    if '款号' in df_dept.columns:
        style_stats = df_dept.groupby(['款号'])['交货数'].agg(['sum', 'mean', 'count']).round(2)
        style_stats.columns = ['總交貨數', '平均交貨數', '交易次數']
        style_stats = style_stats.sort_values('總交貨數', ascending=False)
    else:
        style_stats = None
    
    # 基本統計：活躍天數 = 有交貨的天數
    active_days = df_dept['开单日期'].dt.date.nunique()
    total_delivery = df_dept['交货数'].sum()
    daily_avg = total_delivery / active_days if active_days > 0 else 0
    
    print(f"\n   📊 {dept} 基本統計:")
    print(f"      - 有交貨記錄筆數: {len(df_dept)} 筆")
    print(f"      - 有交貨天數（活躍天數）: {active_days} 天")
    print(f"      - 總交貨量: {total_delivery:,.0f} 件")
    print(f"      - 日均產能（基於有交貨天數）: {daily_avg:.0f} 件/天")
    
    # 執行預測
    result = forecast_dept(df_dept, dept, forecast_days=30)
    
    if result is None:
        print(f"   ⚠️ {dept}: 資料不足 (僅{len(df_dept)}筆)，無法進行可靠預測")
        continue
    
    all_results.append(result)
    
    # 準備預測結果表格
    daily_total = df_dept.groupby('开单日期')['交货数'].sum()
    # 預測日期從最後一個有交貨的日期後一天開始
    forecast_dates = pd.date_range(start=daily_total.index.max() + timedelta(days=1), periods=30, freq='D')
    forecast_table = pd.DataFrame({
        '預測日期': forecast_dates,
        '預測交貨數': [round(v) for v in result['预测值']]
    })
    
    # 交貨次數分布
    bins = [0, 500, 1000, 2000, 5000, 10000, 50000, 100000]
    labels = ['<500', '500-1000', '1000-2000', '2000-5000', '5000-10000', '10000-50000', '>50000']
    df_dept['交貨區間'] = pd.cut(df_dept['交货数'], bins=bins, labels=labels)
    count_dist = df_dept['交貨區間'].value_counts().sort_index().reset_index()
    count_dist.columns = ['交貨區間(件)', '次數']
    
    # 生成檔名（去除特殊字符）
    safe_name = dept.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('組', '组')
    excel_file = os.path.join(output_folder, f'{safe_name}_生产报告.xlsx')
    
    # ============================================
    # 4. 匯出該部門的 Excel 報告
    # ============================================
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 部門概況
            summary_df = pd.DataFrame([{
                '部門名稱': result['部门'],
                '預測模型': result['模型'],
                '資料筆數': len(df_dept),
                '有交貨天數(活躍天數)': result['活跃天数'],
                '歷史總產能(件)': result['历史总产能'],
                '歷史日均產能(件/天)': result['历史日均产能'],
                '預測30天總產能(件)': result['预测总产能'],
                '預測30天日均產能(件/天)': result['预测日均产能'],
                '成長趨勢(%)': result['增长趋势'],
                '週期性模式': '是' if result['has_seasonality'] else '否',
                '備註': '日均產能 = 總交貨量 ÷ 有交貨天數（不含交貨為0的天數）'
            }])
            summary_df.to_excel(writer, sheet_name='部門概況', index=False)
            
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
            
            # 每日預測明細
            forecast_table.to_excel(writer, sheet_name='每日預測明細', index=False)
            
            # 基本統計
            basic_stats = pd.DataFrame({
                '統計項目': [
                    '總交貨件數', 
                    '總交易次數', 
                    '平均每次交貨', 
                    '最大單次交貨', 
                    '最小單次交貨', 
                    '標準差',
                    '有交貨天數(活躍天數)',
                    '歷史日均產能(基於有交貨天數)',
                    '交貨為0的天數',
                    '計算說明'
                ],
                '數值': [
                    f"{df_dept['交货数'].sum():,}",
                    len(df_dept),
                    f"{df_dept['交货数'].mean():.0f}",
                    f"{df_dept['交货数'].max():,}",
                    f"{df_dept['交货数'].min():,}",
                    f"{df_dept['交货数'].std():.0f}",
                    active_days,
                    f"{daily_avg:.0f}",
                    '不納入計算',
                    '只統計交貨數 > 0 的記錄，交貨為0的天數不計入活躍天數'
                ]
            })
            basic_stats.to_excel(writer, sheet_name='基本統計', index=False)
            
            # 交貨次數分布
            count_dist.to_excel(writer, sheet_name='交貨次數分布', index=False)
        
        print(f"   ✅ Excel 報告已儲存: {excel_file}")
        
    except Exception as e:
        print(f"   ⚠️ Excel 匯出失敗: {e}")
    
    # ============================================
    # 5. 繪製該部門的詳細預測圖
    # ============================================
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 圖1: 每日交貨趨勢（只顯示有交貨的日期）
        daily_total = df_dept.groupby('开单日期')['交货数'].sum()
        # 計算移動平均（基於數據點順序）
        daily_values = daily_total.values
        ma_7 = pd.Series(daily_values).rolling(window=7, min_periods=1).mean()
        
        axes[0,0].plot(range(len(daily_total)), daily_total.values, 'b-o', linewidth=1, markersize=3, alpha=0.7, label='每日交貨')
        axes[0,0].plot(range(len(ma_7)), ma_7.values, 'r-', linewidth=2, label='7日移動平均(基於有交貨日)')
        axes[0,0].set_title(f'{dept} - 每日交貨趨勢\n(共{len(daily_total)}個有交貨日期)', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('有交貨日期序號')
        axes[0,0].set_ylabel('交貨數 (件)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 圖2: 每月交貨趨勢（加上數量標籤）
        monthly_total = df_dept.groupby(df_dept['开单日期'].dt.to_period('M'))['交货数'].sum()
        months = range(len(monthly_total))
        month_labels = [str(m) for m in monthly_total.index]
        
        bars = axes[0,1].bar(months, monthly_total.values, color='skyblue', edgecolor='navy')
        axes[0,1].set_title(f'{dept} - 每月交貨趨勢', fontsize=12, fontweight='bold')
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
        bins = [0, 500, 1000, 2000, 5000, 10000, 50000, 100000]
        labels = ['<500', '500-1000', '1000-2000', '2000-5000', '5000-10000', '10000-50000', '>50000']
        df_dept['交貨區間'] = pd.cut(df_dept['交货数'], bins=bins, labels=labels)
        count_dist_plot = df_dept['交貨區間'].value_counts().sort_index()
        
        bars = axes[1,0].bar(range(len(count_dist_plot)), count_dist_plot.values, color='lightcoral', edgecolor='darkred')
        axes[1,0].set_title(f'{dept} - 交貨次數分布', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('交貨數量區間 (件)')
        axes[1,0].set_ylabel('次數')
        axes[1,0].set_xticks(range(len(count_dist_plot)))
        axes[1,0].set_xticklabels(count_dist_plot.index, rotation=45)
        for i, (idx, val) in enumerate(count_dist_plot.items()):
            axes[1,0].text(i, val + 0.5, str(val), ha='center', va='bottom')
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # 圖4: 產能預測
        last_30_values = daily_total.tail(30)
        forecast_values = result['预测值']
        
        if len(last_30_values) > 0:
            axes[1,1].plot(range(len(last_30_values)), last_30_values.values, 'b-o', linewidth=2, markersize=4, label='歷史數據 (過去30個有交貨日)')
        
        axes[1,1].plot(range(len(last_30_values), len(last_30_values) + 30), forecast_values, 'r--o', linewidth=2, markersize=4, label='未來30天預測')
        
        hist_avg = result['历史日均产能']
        pred_avg = result['预测日均产能']
        axes[1,1].axhline(y=hist_avg, color='blue', linestyle=':', alpha=0.5, label=f'歷史日均: {hist_avg:.0f}件')
        axes[1,1].axhline(y=pred_avg, color='red', linestyle=':', alpha=0.5, label=f'預測日均: {pred_avg:.0f}件')
        
        axes[1,1].set_title(f'{dept} - 產能預測 (未來30天)\n預測30天總產能: {result["预测总产能"]:,.0f}件 | 成長趨勢: {result["增长趋势"]:+.1f}%', 
                           fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('有交貨日期序號')
        axes[1,1].set_ylabel('交貨數 (件)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        image_file = os.path.join(output_folder, f'{safe_name}_产能预测.png')
        plt.savefig(image_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ 預測圖已儲存: {image_file}")
        
    except Exception as e:
        print(f"   ⚠️ 圖表繪製失敗: {e}")
        plt.close()
    
    # 顯示預測結果摘要
    print(f"\n   📊 {dept} 預測結果摘要:")
    print(f"      - 有交貨記錄筆數: {len(df_dept)} 筆")
    print(f"      - 有交貨天數(活躍天數): {active_days} 天")
    print(f"      - 歷史總產能: {result['历史总产能']:,.0f} 件")
    print(f"      - 歷史日均: {result['历史日均产能']:,.0f} 件/天 (基於有交貨天數)")
    print(f"      - 預測日均: {result['预测日均产能']:,.0f} 件/天")
    print(f"      - 預測30天總產能: {result['预测总产能']:,.0f} 件")
    print(f"      - 成長趨勢: {result['增长趋势']:+.1f}%")
    print(f"      - 使用模型: {result['模型']}")
    if result['has_seasonality']:
        print(f"      - 週期性: 偵測到星期效應")

# ============================================
# 6. 生成匯總報告
# ============================================
print("\n" + "=" * 70)
print("📊 生成匯總報告")
print("=" * 70)

if all_results:
    # 匯總所有部門預測結果
    summary_all = pd.DataFrame([{
        '部門': r['部门'],
        '預測模型': r['模型'],
        '有交貨筆數': len(df_raw[df_raw['发出部门'] == r['部门']]),
        '有交貨天數(活躍天數)': r['活跃天数'],
        '歷史總產能(件)': r['历史总产能'],
        '歷史日均產能(件/天)': r['历史日均产能'],
        '預測30天總產能(件)': r['预测总产能'],
        '預測30天日均產能(件/天)': r['预测日均产能'],
        '成長趨勢(%)': r['增长趋势'],
        '週期性模式': '是' if r['has_seasonality'] else '否'
    } for r in all_results])
    
    summary_all = summary_all.sort_values('預測30天總產能(件)', ascending=False)
    
    # 儲存匯總報告
    summary_file = os.path.join(output_folder, '00_各部门汇总报告.xlsx')
    try:
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            summary_all.to_excel(writer, sheet_name='各部門預測匯總', index=False)
            
            # 各部門歷史總統計
            dept_summary = df_raw.groupby('发出部门').agg({
                '交货数': ['sum', 'mean', 'count'],
                '开单日期': lambda x: f"{x.min().strftime('%Y-%m-%d')} ~ {x.max().strftime('%Y-%m-%d')}"
            }).round(2)
            dept_summary.columns = ['總交貨數', '平均交貨數', '交易次數', '日期範圍']
            dept_summary.to_excel(writer, sheet_name='各部門歷史統計')
            
            # 整體時間維度統計
            df_raw['年月'] = df_raw['开单日期'].dt.strftime('%Y-%m')
            monthly_all = df_raw.groupby(['年月', '发出部门'])['交货数'].sum().reset_index()
            monthly_pivot = monthly_all.pivot(index='年月', columns='发出部门', values='交货数').fillna(0)
            monthly_pivot.to_excel(writer, sheet_name='各部門月報')
            
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
        
        print(f"✅ 匯總報告已儲存: {summary_file}")
    except Exception as e:
        print(f"⚠️ 匯總報告儲存失敗: {e}")

# ============================================
# 7. 最終摘要報告
# ============================================
print("\n" + "=" * 70)
print("📋 各部門產能預測摘要報告")
print("=" * 70)

print(f"\n📊 整體統計:")
print(f"   - 部門總數: {len(departments)}")
print(f"   - 成功預測部門數: {len(all_results)}")
print(f"   - 資料總筆數(僅交貨>0): {len(df_raw):,}")
print(f"   - 歷史總交貨量: {df_raw['交货数'].sum():,} 件")

print(f"\n📌 重要說明:")
print("   - 本報告只統計交貨數 > 0 的記錄")
print("   - 交貨為0的天數不納入計算")
print("   - 活躍天數 = 有交貨的天數")
print("   - 日均產能 = 總交貨量 ÷ 有交貨天數")

if all_results:
    total_forecast = sum([r['预测总产能'] for r in all_results])
    print(f"\n🔮 各部門預測30天總產能合計: {total_forecast:,.0f} 件")
    print(f"📈 預測30天整體日均產能: {total_forecast/30:.0f} 件/天")
    
    print("\n📌 各部門預測明細 (依預測產能排序):")
    sorted_results = sorted(all_results, key=lambda x: x['预测总产能'], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        growth_symbol = "📈" if r['增长趋势'] > 3 else "📉" if r['增长趋势'] < -3 else "➡️"
        print(f"   {i}. {r['部门']}: {r['预测总产能']:>12,} 件 (日均: {r['预测日均产能']:>6,.0f} | 活躍天數: {r['活跃天数']:>4}天 | 趨勢: {r['增长趋势']:>+6.1f}% {growth_symbol})")

print("\n" + "=" * 70)
print("📁 輸出檔案結構:")
print(f"\n{output_folder}/")
print("   ├── 00_各部门汇总报告.xlsx     (所有部門匯總，含計算說明)")
print("   ├── 車縫2组_生产报告.xlsx      (車縫2組完整報告)")
print("   ├── 車縫2组_产能预测.png       (車縫2組預測圖)")
print("   ├── 鈕壓_生产报告.xlsx          (鈕壓完整報告)")
print("   ├── 鈕壓_产能预测.png           (鈕壓預測圖)")
print("   └── ... (其他部門)")
print("=" * 70)

print("\n✨ 所有部門分析完成！")
print(f"📁 報告已儲存至: {os.path.abspath(output_folder)}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
from math import sqrt
warnings.filterwarnings('ignore')

# 更新款號列表 - 與目標Excel一致
items = ['P12857-1', 'P12857-4', 'P12857-TD6', 'P12857-3A', 'P12857-TM10', 'P12857-TR2']
test_days = 30
forecast_days = 30

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出資料夾
output_folder = f'车缝组_多模型预测对比报告_{datetime.now().strftime("%Y%m%d")}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"📁 創建資料夾: {output_folder}")

# ============================================
# 1. 讀取 Excel 資料
# ============================================
print("=" * 80)
print(f"📊 車縫組 - 多模型預測對比報告 (款號: {', '.join(items)})")
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

# 資料清理
df_raw['开单日期'] = pd.to_datetime(df_raw['开单日期'])
df_raw = df_raw.dropna(subset=['开单日期', '发出部门', '交货数'])
df_raw = df_raw[df_raw['交货数'] > 0]
df_raw = df_raw.sort_values('开单日期')

# ============================================
# 2. 篩選條件
# ============================================
print("\n" + "=" * 80)
print("🔍 篩選條件:")
print("   - 發出部門 包含「車縫」")
print(f"   - 款號: {items}")
print("   - 排除星期日（星期天不預測）")
print("=" * 80)

sewing_mask = df_raw['发出部门'].str.contains('車縫', na=False)
df_sewing = df_raw[sewing_mask].copy()

if '款号' in df_sewing.columns:
    mask = df_sewing['款号'].isin(items)
    df_filtered = df_sewing[mask].copy()
    print(f"\n✅ 篩選完成: {len(df_filtered)} 筆資料")
    print(f"   - 款號: {df_filtered['款号'].unique().tolist()}")
else:
    df_filtered = df_sewing.copy()
    print(f"\n⚠️ 無款號欄位，僅篩選車縫部門")

if len(df_filtered) == 0:
    print(f"\n❌ 沒有符合條件的資料！")
    exit()

# 添加星期幾資訊
df_filtered['星期幾'] = df_filtered['开单日期'].dt.dayofweek
df_filtered['星期'] = df_filtered['开单日期'].dt.dayofweek.map({
    0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'
})

# 每日彙總（排除星期日用於訓練）
df_filtered_no_sunday = df_filtered[df_filtered['星期幾'] != 6].copy()
daily_total = df_filtered_no_sunday.groupby('开单日期')['交货数'].sum()

# 包含所有日期的每日彙總（用於統計）
daily_total_all = df_filtered.groupby('开单日期')['交货数'].sum()

print(f"\n📊 歷史資料統計:")
print(f"   - 原始有交貨天數: {len(daily_total_all)} 天")
print(f"   - 排除星期日後天數: {len(daily_total)} 天")
print(f"   - 總交貨量: {daily_total.sum():,.0f} 件")
print(f"   - 日均產能(僅工作日): {daily_total.mean():.0f} 件/天")

# ============================================
# 3. 定義預測函數
# ============================================

def get_all_forecast_dates(start_date, num_days):
    """獲取包含星期日的完整日期列表（共num_days天）"""
    return [start_date + timedelta(days=i) for i in range(num_days)]

def simple_moving_average(data, window, forecast_days=30):
    """簡單移動平均"""
    ma = data.rolling(window=window, min_periods=1).mean()
    last_ma = ma.iloc[-1]
    forecast = [last_ma] * forecast_days
    
    return {
        'name': f'SMA-{window}',
        'forecast': forecast,
        'params': {'window': window},
        'last_value': last_ma,
        'model_type': '移动平均',
        'history_ma': ma  # 保存歷史移動平均
    }

def weighted_moving_average(data, window, forecast_days=30):
    """加權移動平均"""
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    
    last_n = data.iloc[-window:].values if len(data) >= window else data.values
    if len(last_n) >= window:
        wma = np.sum(last_n * weights[-len(last_n):])
    else:
        wma = data.mean()
    
    forecast = [wma] * forecast_days
    
    return {
        'name': f'WMA-{window}',
        'forecast': forecast,
        'params': {'window': window, 'weights': 'linear'},
        'last_value': wma,
        'model_type': '加权移动平均'
    }

def exponential_smoothing(data, alpha, forecast_days=30):
    """指數平滑"""
    result = [data.iloc[0]]
    for t in range(1, len(data)):
        result.append(alpha * data.iloc[t] + (1 - alpha) * result[-1])
    
    last_smooth = result[-1]
    forecast = [last_smooth] * forecast_days
    
    return {
        'name': f'ES-α={alpha}',
        'forecast': forecast,
        'params': {'alpha': alpha},
        'last_value': last_smooth,
        'model_type': '指数平滑'
    }

def double_exponential_smoothing(data, alpha, beta, forecast_days=30):
    """Holt's 線性指數平滑"""
    n = len(data)
    level = [data.iloc[0]]
    trend = [data.iloc[1] - data.iloc[0] if n > 1 else 0]
    
    for t in range(1, n):
        new_level = alpha * data.iloc[t] + (1 - alpha) * (level[-1] + trend[-1])
        new_trend = beta * (new_level - level[-1]) + (1 - beta) * trend[-1]
        level.append(new_level)
        trend.append(new_trend)
    
    forecast = []
    last_level = level[-1]
    last_trend = trend[-1]
    for i in range(1, forecast_days + 1):
        forecast.append(last_level + i * last_trend)
    
    return {
        'name': f'Holt-α={alpha},β={beta}',
        'forecast': forecast,
        'params': {'alpha': alpha, 'beta': beta},
        'last_value': forecast[0] if forecast else data.mean(),
        'model_type': '双指数平滑'
    }

def seasonal_naive(data, period=7, forecast_days=30):
    """季節性朴素預測"""
    if len(data) < period:
        return None
    
    forecast = []
    for i in range(forecast_days):
        idx = -(period - (i % period)) if (i % period) != 0 else -period
        if abs(idx) <= len(data):
            pred = data.iloc[idx]
        else:
            pred = data.mean()
        forecast.append(pred)
    
    return {
        'name': f'Seasonal-周期{period}天',
        'forecast': forecast,
        'params': {'period': period},
        'last_value': forecast[0],
        'model_type': '季节性预测'
    }

def evaluate_model(data, model_func, test_days=test_days, **kwargs):
    """評估模型"""
    if len(data) < test_days + 5:
        return None
    
    train_data = data.iloc[:-test_days]
    test_data = data.iloc[-test_days:]
    
    try:
        if 'alpha' in kwargs and 'beta' in kwargs:
            result = model_func(train_data, forecast_days=test_days, **kwargs)
        elif 'alpha' in kwargs:
            result = model_func(train_data, forecast_days=test_days, **kwargs)
        elif 'window' in kwargs:
            result = model_func(train_data, forecast_days=test_days, **kwargs)
        else:
            result = model_func(train_data, forecast_days=test_days, **kwargs)
        
        if result is None:
            return None
        
        predictions = result['forecast'][:test_days]
        
        mae = np.mean(np.abs(test_data.values - predictions))
        rmse = sqrt(np.mean((test_data.values - predictions) ** 2))
        mape = np.mean(np.abs((test_data.values - predictions) / test_data.values)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': predictions,
            'actual': test_data.values
        }
    except Exception as e:
        print(f"   ⚠️ 模型評估失敗: {e}")
        return None

# ============================================
# 4. 定義所有要測試的模型
# ============================================
print("\n" + "=" * 80)
print("🤖 初始化預測模型 (排除星期日)")
print("=" * 80)

models_config = [
    {'func': simple_moving_average, 'params': {'window': 3}, 'name': 'SMA-3'},
    {'func': simple_moving_average, 'params': {'window': 7}, 'name': 'SMA-7'},
    {'func': simple_moving_average, 'params': {'window': 15}, 'name': 'SMA-15'},
    {'func': simple_moving_average, 'params': {'window': 30}, 'name': 'SMA-30'},
    
    {'func': weighted_moving_average, 'params': {'window': 7}, 'name': 'WMA-7'},
    {'func': weighted_moving_average, 'params': {'window': 15}, 'name': 'WMA-15'},
    
    {'func': exponential_smoothing, 'params': {'alpha': 0.3}, 'name': 'ES-α=0.3'},
    {'func': exponential_smoothing, 'params': {'alpha': 0.5}, 'name': 'ES-α=0.5'},
    {'func': exponential_smoothing, 'params': {'alpha': 0.7}, 'name': 'ES-α=0.7'},
    
    {'func': double_exponential_smoothing, 'params': {'alpha': 0.5, 'beta': 0.3}, 'name': 'Holt-0.5,0.3'},
    {'func': double_exponential_smoothing, 'params': {'alpha': 0.6, 'beta': 0.2}, 'name': 'Holt-0.6,0.2'},
    
    {'func': seasonal_naive, 'params': {'period': 7}, 'name': 'Seasonal-7'},
    {'func': seasonal_naive, 'params': {'period': 15}, 'name': 'Seasonal-15'},
    {'func': seasonal_naive, 'params': {'period': 30}, 'name': 'Seasonal-30'},
]

all_models = []
print("\n📊 訓練模型中...")

for config in models_config:
    try:
        if 'window' in config['params']:
            result = config['func'](daily_total, forecast_days=forecast_days, **config['params'])
        elif 'alpha' in config['params'] and 'beta' in config['params']:
            result = config['func'](daily_total, forecast_days=forecast_days, **config['params'])
        elif 'alpha' in config['params']:
            result = config['func'](daily_total, forecast_days=forecast_days, **config['params'])
        elif 'period' in config['params']:
            result = config['func'](daily_total, forecast_days=forecast_days, **config['params'])
        else:
            continue
        
        if result is not None:
            evaluation = evaluate_model(daily_total, config['func'], test_days=test_days, **config['params'])
            
            result['evaluation'] = evaluation
            result['config'] = config
            all_models.append(result)
            print(f"   ✅ {result['name']} - 訓練完成")
    except Exception as e:
        print(f"   ⚠️ {config['name']} 訓練失敗: {e}")

print(f"\n✅ 成功訓練 {len(all_models)} 個模型")

# ============================================
# 5. 應用星期日過濾到預測結果
# ============================================
print("\n" + "=" * 80)
print("📅 應用星期日過濾（星期日預測值設為0）")
print("=" * 80)

last_date = daily_total.index.max()
start_forecast_date = last_date + timedelta(days=1)

# 生成完整的30天預測日期（包含星期日）
all_forecast_dates = get_all_forecast_dates(start_forecast_date, forecast_days)

# 計算星期日數量
sunday_count = sum(1 for d in all_forecast_dates if d.weekday() == 6)
workday_count = forecast_days - sunday_count

print(f"   - 預測期間: {all_forecast_dates[0].strftime('%Y-%m-%d')} ~ {all_forecast_dates[-1].strftime('%Y-%m-%d')}")
print(f"   - 星期日天數: {sunday_count} 天 (預測值設為0)")
print(f"   - 工作日天數: {workday_count} 天")

for model in all_models:
    filtered_forecast = []
    workday_idx = 0
    for forecast_date in all_forecast_dates:
        if forecast_date.weekday() == 6:
            filtered_forecast.append(0)
        else:
            if workday_idx < len(model['forecast']):
                filtered_forecast.append(model['forecast'][workday_idx])
                workday_idx += 1
            else:
                filtered_forecast.append(0)
    
    model['original_forecast'] = model['forecast']
    model['forecast'] = filtered_forecast

print(f"   ✅ 已對 {len(all_models)} 個模型應用星期日過濾")

# ============================================
# 6. 模型對比分析
# ============================================
print("\n" + "=" * 80)
print("📊 模型對比分析")
print("=" * 80)

comparison_data = []
for model in all_models:
    pred_total = sum(model['forecast'])
    pred_daily = pred_total / forecast_days
    hist_daily = daily_total.mean()
    change_pct = ((pred_daily - hist_daily) / hist_daily * 100) if hist_daily > 0 else 0
    
    workday_forecast = [v for v in model['forecast'] if v > 0]
    workday_avg = np.mean(workday_forecast) if workday_forecast else 0
    
    row = {
        '模型名稱': model['name'],
        '模型類型': model['model_type'],
        '參數': str(model['params']),
        '歷史日均(件)': round(hist_daily, 0),
        '預測日均(含周日)': round(pred_daily, 0),
        '預測工作日日均(件)': round(workday_avg, 0),
        '預測30天總產能(件)': round(pred_total, 0),
        '產能變化率(%)': round(change_pct, 1),
    }
    
    if model.get('evaluation'):
        row['驗證MAE'] = round(model['evaluation']['mae'], 0)
        row['驗證RMSE'] = round(model['evaluation']['rmse'], 0)
        row['驗證MAPE(%)'] = round(model['evaluation']['mape'], 1)
    
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
if '驗證MAE' in comparison_df.columns:
    comparison_df = comparison_df.sort_values('驗證MAE')

best_model = min(all_models, key=lambda x: x.get('evaluation', {}).get('mae', float('inf')))

print(f"\n🏆 最佳模型: {best_model['name']}")
if best_model.get('evaluation'):
    print(f"   - MAE: {best_model['evaluation']['mae']:.0f} 件")
    print(f"   - MAPE: {best_model['evaluation']['mape']:.1f}%")

# ============================================
# 7. 計算移動平均歷史數據（用於最佳模型）
# ============================================
# 獲取最佳模型的窗口（如果是移動平均類）
best_window = None
if 'window' in best_model.get('params', {}):
    best_window = best_model['params']['window']
    
    # 計算歷史移動平均
    ma_history = daily_total.rolling(window=best_window, min_periods=1).mean()
    ma_history_df = pd.DataFrame({
        '日期': ma_history.index,
        '實際交貨數': daily_total.values,
        f'{best_window}天移動平均': ma_history.values
    })
else:
    ma_history_df = pd.DataFrame({
        '日期': daily_total.index,
        '實際交貨數': daily_total.values,
        '移動平均': daily_total.rolling(window=7, min_periods=1).mean().values
    })

# ============================================
# 8. 生成Excel報告（與目標格式一致）
# ============================================
print("\n" + "=" * 80)
print("📊 生成Excel綜合報告（與目標格式一致）")
print("=" * 80)

excel_file = os.path.join(output_folder, f'车缝组_多模型预测报告.xlsx')

try:
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        # ===== Sheet 1: 預測摘要 (使用最佳模型) =====
        pred_total = sum(best_model['forecast'])
        pred_daily_avg = pred_total / forecast_days
        hist_daily_avg = daily_total.mean()
        change_pct = ((pred_daily_avg - hist_daily_avg) / hist_daily_avg * 100) if hist_daily_avg > 0 else 0
        
        # 獲取窗口參數
        window_str = ''
        if 'window' in best_model.get('params', {}):
            window_str = f"{best_model['params']['window']}天"
        elif 'alpha' in best_model.get('params', {}):
            window_str = f"α={best_model['params']['alpha']}"
        else:
            window_str = best_model['name']
        
        summary_data = {
            '預測模型': [best_model['name']],
            '移動平均窗口': [window_str],
            '歷史有交貨天數': [len(daily_total)],
            '歷史總產能(件)': [daily_total.sum()],
            '歷史日均產能(件/天)': [round(hist_daily_avg, 0)],
            '最後移動平均值(件/天)': [round(best_model['last_value'], 0)],
            '預測30天總產能(件)': [round(pred_total, 0)],
            '預測30天日均產能(件/天)': [round(pred_daily_avg, 0)],
            '產能變化率(%)': [round(change_pct, 1)],
            '預測起始日': [all_forecast_dates[0].strftime('%Y-%m-%d')],
            '預測結束日': [all_forecast_dates[-1].strftime('%Y-%m-%d')],
            '備註': ['只計算有交貨的日期，星期日不生產']
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='預測摘要', index=False)
        print(f"   ✅ 工作表 '預測摘要' 已儲存")
        
        # ===== Sheet 2: 每日預測明細 (最佳模型) =====
        daily_forecast_df = pd.DataFrame({
            '預測日期': all_forecast_dates,
            '預測交貨數': [round(v) for v in best_model['forecast']]
        })
        daily_forecast_df.to_excel(writer, sheet_name='每日預測明細', index=False)
        print(f"   ✅ 工作表 '每日預測明細' 已儲存")
        
        # ===== Sheet 3: 移動平均歷史數據 =====
        ma_history_df.to_excel(writer, sheet_name='移動平均歷史數據', index=False)
        print(f"   ✅ 工作表 '移動平均歷史數據' 已儲存")
        
        # ===== Sheet 4: 原始資料 =====
        original_data_cols = ['开单日期', '制单号', 'MO.NO', '款号', '制单数', 
                               '发出部门', '接收部门', '交货数']
        available_cols = [col for col in original_data_cols if col in df_filtered.columns]
        original_data_df = df_filtered[available_cols].copy()
        original_data_df = original_data_df.sort_values('开单日期')
        original_data_df.to_excel(writer, sheet_name='原始資料', index=False)
        print(f"   ✅ 工作表 '原始資料' 已儲存")
        
        # ===== Sheet 5: 統計分析 =====
        stats_data = {
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
                f"{daily_total.sum():,}",
                len(df_filtered),
                round(df_filtered['交货数'].mean(), 0),
                df_filtered['交货数'].max(),
                df_filtered['交货数'].min(),
                round(df_filtered['交货数'].std(), 0),
                len(daily_total),
                round(daily_total.mean(), 0)
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='統計分析', index=False)
        print(f"   ✅ 工作表 '統計分析' 已儲存")
        
        # ===== Sheet 6: 款號統計 =====
        if '款号' in df_filtered.columns:
            style_stats = df_filtered.groupby('款号').agg({
                '交货数': ['sum', 'mean', 'count']
            }).round(0)
            style_stats.columns = ['總交貨數', '平均交貨數', '交易次數']
            style_stats = style_stats.reset_index()
            style_stats.to_excel(writer, sheet_name='款號統計', index=False)
            print(f"   ✅ 工作表 '款號統計' 已儲存")
        
        # ===== Sheet 7: 部門統計 =====
        dept_stats = df_filtered.groupby('发出部门').agg({
            '交货数': ['sum', 'mean', 'count']
        }).round(0)
        dept_stats.columns = ['總交貨數', '平均交貨數', '交易次數']
        dept_stats = dept_stats.reset_index()
        dept_stats.to_excel(writer, sheet_name='部門統計', index=False)
        print(f"   ✅ 工作表 '部門統計' 已儲存")
        
        # 額外添加：多模型對比表
        comparison_df.to_excel(writer, sheet_name='多模型對比', index=False)
        print(f"   ✅ 工作表 '多模型對比' 已儲存")
        
        # 額外添加：所有模型預測匯總
        all_forecasts = pd.DataFrame({'預測日期': all_forecast_dates})
        for model in all_models[:10]:  # 最多顯示10個模型
            all_forecasts[model['name']] = [round(v) for v in model['forecast']]
        all_forecasts.to_excel(writer, sheet_name='所有模型預測匯總', index=False)
        print(f"   ✅ 工作表 '所有模型預測匯總' 已儲存")
    
    print(f"\n✅ Excel報告已儲存: {excel_file}")

except Exception as e:
    print(f"   ⚠️ Excel儲存失敗: {e}")

# ============================================
# 9. 生成多模型對比圖表
# ============================================
print("\n" + "=" * 80)
print("📈 生成多模型對比圖表")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'車縫組 - 多模型預測對比分析 (款號: {", ".join(items)})', fontsize=16, fontweight='bold')

# 1. 歷史數據 + 最佳模型預測
ax = axes[0, 0]
show_days = min(60, len(daily_total))
ax.plot(daily_total.index[-show_days:], daily_total.values[-show_days:], 'b-o', linewidth=1.5, markersize=3, label='歷史實際交貨')

non_sunday_dates = [all_forecast_dates[i] for i in range(forecast_days) if all_forecast_dates[i].weekday() != 6]
non_sunday_values = [best_model['forecast'][i] for i in range(forecast_days) if all_forecast_dates[i].weekday() != 6]
sunday_dates = [all_forecast_dates[i] for i in range(forecast_days) if all_forecast_dates[i].weekday() == 6]

ax.plot(non_sunday_dates, non_sunday_values, 'r--o', linewidth=2, markersize=4, label=f'最佳模型: {best_model["name"]}')
if sunday_dates:
    ax.scatter(sunday_dates, [0]*len(sunday_dates), color='gray', marker='x', s=50, label='星期日(不生產)')

ax.axhline(y=daily_total.mean(), color='blue', linestyle=':', alpha=0.5, label=f'歷史日均: {daily_total.mean():.0f}')
ax.axhline(y=sum(best_model['forecast'])/forecast_days, color='red', linestyle=':', alpha=0.5, label=f'預測日均: {sum(best_model["forecast"])/forecast_days:.0f}')
ax.set_title(f'最佳模型預測 - {best_model["name"]}\nMAE: {best_model.get("evaluation", {}).get("mae", "N/A"):.0f} | MAPE: {best_model.get("evaluation", {}).get("mape", "N/A"):.1f}%', fontsize=11, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('交貨數 (件)')
ax.legend(loc='upper left', fontsize=8)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# 2. 模型誤差對比
ax = axes[0, 1]
models_with_mae = [(m['name'], m.get('evaluation', {}).get('mae', 0)) for m in all_models if m.get('evaluation')]
models_with_mae.sort(key=lambda x: x[1])
if models_with_mae:
    names = [m[0][:15] for m in models_with_mae[:10]]
    maes = [m[1] for m in models_with_mae[:10]]
    bars = ax.barh(names, maes, color='skyblue', edgecolor='navy')
    ax.set_title('模型誤差對比 (MAE - 越低越好)', fontsize=11, fontweight='bold')
    ax.set_xlabel('平均絕對誤差 (件)')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, maes):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f'{val:.0f}', va='center', fontsize=9)

# 3. 模型預測範圍對比
ax = axes[0, 2]
all_predictions = [m['forecast'] for m in all_models]
pred_array = np.array(all_predictions)
pred_mean = pred_array.mean(axis=0)
pred_std = pred_array.std(axis=0)
pred_min = pred_array.min(axis=0)
pred_max = pred_array.max(axis=0)

ax.fill_between(range(forecast_days), pred_min, pred_max, alpha=0.3, color='gray', label='預測範圍(最小-最大)')
ax.plot(range(forecast_days), pred_mean, 'b-', linewidth=2, label='平均預測')
ax.plot(range(forecast_days), [daily_total.mean()] * forecast_days, 'r--', label='歷史日均線')
ax.set_title(f'多模型預測範圍 (共{len(all_models)}個模型)', fontsize=11, fontweight='bold')
ax.set_xlabel('預測天數')
ax.set_ylabel('預測交貨數 (件)')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 最佳3個模型預測對比
ax = axes[1, 0]
top_3 = sorted(all_models, key=lambda x: x.get('evaluation', {}).get('mae', float('inf')))[:3]
colors = ['red', 'green', 'orange']
for i, model in enumerate(top_3):
    non_sunday_vals = [model['forecast'][j] for j in range(forecast_days) if all_forecast_dates[j].weekday() != 6]
    ax.plot(non_sunday_dates, non_sunday_vals, color=colors[i], linewidth=2, 
            label=f"{model['name']} (MAE:{model.get('evaluation',{}).get('mae',0):.0f})")
ax.plot(daily_total.index[-30:], daily_total.values[-30:], 'b-o', linewidth=1.5, markersize=3, label='歷史數據')
ax.set_title('最佳3個模型預測對比', fontsize=11, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('交貨數 (件)')
ax.legend(loc='upper left', fontsize=8)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# 5. 各款號交貨量統計
ax = axes[1, 1]
if '款号' in df_filtered.columns:
    style_totals = df_filtered.groupby('款号')['交货数'].sum().sort_values()
    colors_bar = plt.cm.Set3(range(len(style_totals)))
    bars = ax.barh(style_totals.index, style_totals.values, color=colors_bar, edgecolor='black')
    ax.set_title('各款號總交貨量統計', fontsize=11, fontweight='bold')
    ax.set_xlabel('總交貨數 (件)')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, style_totals.values):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2, f'{val:,.0f}', va='center', fontsize=9)

# 6. 各星期平均交貨量分析
ax = axes[1, 2]
weekday_avg = df_filtered.groupby('星期')['交货数'].mean()
weekday_order = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
weekday_avg = weekday_avg.reindex(weekday_order)
colors_bar = ['steelblue'] * 6 + ['lightcoral']
bars = ax.bar(weekday_order, weekday_avg.values, color=colors_bar, edgecolor='black')
ax.set_title('各星期平均交貨量分析', fontsize=11, fontweight='bold')
ax.set_xlabel('星期')
ax.set_ylabel('平均交貨數 (件)')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, weekday_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '00_多模型预测对比总图.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   ✅ 多模型對比總圖已儲存")

# ============================================
# 10. 生成詳細預測圖（最佳3個模型）
# ============================================
print("\n📊 生成各模型詳細預測圖...")

for i, model in enumerate(top_3):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    show_days = min(60, len(daily_total))
    ax.plot(daily_total.index[-show_days:], daily_total.values[-show_days:], 
            'b-o', linewidth=1.5, markersize=3, label='歷史交貨量')
    
    non_sunday_vals = [model['forecast'][j] for j in range(forecast_days) if all_forecast_dates[j].weekday() != 6]
    ax.plot(non_sunday_dates, non_sunday_vals, 'r--o', linewidth=2, markersize=4, label='預測值')
    if sunday_dates:
        ax.scatter(sunday_dates, [0]*len(sunday_dates), color='gray', marker='x', s=50, label='星期日(不生產)')
    
    hist_std = daily_total.std()
    upper = [model['forecast'][j] + hist_std for j in range(forecast_days)]
    lower = [max(0, model['forecast'][j] - hist_std) for j in range(forecast_days)]
    ax.fill_between(all_forecast_dates, lower, upper, alpha=0.2, color='red', label='±1σ 置信區間')
    
    ax.axhline(y=daily_total.mean(), color='blue', linestyle=':', alpha=0.5, label=f'歷史日均: {daily_total.mean():.0f}')
    ax.axhline(y=sum(model['forecast'])/forecast_days, color='red', linestyle=':', alpha=0.5, 
               label=f'預測日均: {sum(model["forecast"])/forecast_days:.0f}')
    
    workday_avg = np.mean(non_sunday_vals) if non_sunday_vals else 0
    ax.axhline(y=workday_avg, color='orange', linestyle=':', alpha=0.5, 
               label=f'預測工作日日均: {workday_avg:.0f}')
    
    ax.set_title(f'{model["name"]} - {model["model_type"]}\n預測30天總產能: {sum(model["forecast"]):,.0f}件', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('交貨數 (件)')
    ax.legend(loc='upper left', fontsize=8)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    if model.get('evaluation'):
        ev = model['evaluation']
        days = list(range(1, len(ev['predictions']) + 1))
        ax.plot(days, ev['actual'], 'b-o', linewidth=2, markersize=6, label='實際值')
        ax.plot(days, ev['predictions'], 'r--s', linewidth=2, markersize=6, label='預測值')
        
        errors = np.abs(ev['actual'] - ev['predictions'])
        for d, a, p, e in zip(days, ev['actual'], ev['predictions'], errors):
            ax.vlines(d, min(a, p), max(a, p), color='gray', alpha=0.5, linewidth=1)
        
        ax.set_title(f'驗證集表現 (最近{test_days}個工作日)\nMAE: {ev["mae"]:.0f} | MAPE: {ev["mape"]:.1f}%', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('驗證天數')
        ax.set_ylabel('交貨數 (件)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        if non_sunday_vals:
            ax.hist(non_sunday_vals, bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
            ax.axvline(x=np.mean(non_sunday_vals), color='red', linestyle='--', linewidth=2, 
                       label=f'工作日均值: {np.mean(non_sunday_vals):.0f}')
        ax.set_title(f'預測值分布 (僅工作日)', fontsize=12, fontweight='bold')
        ax.set_xlabel('預測交貨數 (件)')
        ax.set_ylabel('頻次')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = model['name'].replace('=', '').replace(',', '_').replace('-', '_')
    plt.savefig(os.path.join(output_folder, f'{safe_name}_详细预测.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

print(f"   ✅ 已生成 {len(top_3)} 個詳細預測圖")

# ============================================
# 11. 最終報告輸出
# ============================================
print("\n" + "=" * 80)
print("📋 多模型預測最終報告")
print("=" * 80)

print(f"\n🏆 最佳模型: {best_model['name']}")
if best_model.get('evaluation'):
    print(f"   - MAE (平均絕對誤差): {best_model['evaluation']['mae']:.0f} 件")
    print(f"   - MAPE (平均絕對百分比誤差): {best_model['evaluation']['mape']:.1f}%")
print(f"   - 預測30天總產能: {sum(best_model['forecast']):,.0f} 件")
print(f"   - 預測日均產能(含周日): {sum(best_model['forecast'])/forecast_days:.0f} 件/天")

workday_forecast = [v for v in best_model['forecast'] if v > 0]
if workday_forecast:
    print(f"   - 預測工作日日均產能: {np.mean(workday_forecast):.0f} 件/天")

print(f"\n📊 多模型統計:")
print(f"   - 模型總數: {len(all_models)}")
print(f"   - 預測日均範圍: {pred_min.mean():.0f} ~ {pred_max.mean():.0f} 件/天")
print(f"   - 預測30天總產能範圍: {pred_min.sum():.0f} ~ {pred_max.sum():.0f} 件")

print(f"\n📁 輸出檔案結構:")
print(f"\n{output_folder}/")
print("   ├── 车缝组_多模型预测报告.xlsx          (Excel報告)")
print("   ├── 00_多模型预测对比总图.png          (6合1對比圖)")
print("   ├── [模型名稱]_详细预测.png            (各模型詳細圖)")
print("   └── ...")

print("\n" + "=" * 80)
print("⚠️ 重要說明:")
print(f"   1. 篩選款號: {', '.join(items)}")
print(f"   2. 使用{test_days}天歷史數據進行模型驗證")
print("   3. 星期日不生產，預測值設為0")
print("   4. 歷史訓練數據已排除星期日")
print("   5. MAE越低代表模型越準確")
print("=" * 80)

print("\n✨ 多模型預測對比報告生成完成！")
print(f"📁 報告已儲存至: {os.path.abspath(output_folder)}")
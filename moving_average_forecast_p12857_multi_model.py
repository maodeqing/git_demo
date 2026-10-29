import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
from math import sqrt
import duckdb as dd
warnings.filterwarnings('ignore')

item = 'P12857'
test_days = 15
forecast_days = 30

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出資料夾
output_folder = f'车缝组_{item}_多模型预测对比报告({test_days=})'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"📁 創建資料夾: {output_folder}")

# ============================================
# 1. 讀取 Excel 資料
# ============================================
print("=" * 80)
print(f"📊 車縫組 - 款號 {item}* 多模型預測對比報告")
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
print(f"   - 款號 以「{item}」開頭")
print("   - 交貨數 > 0")
print("=" * 80)

sewing_mask = df_raw['发出部门'].str.contains('車縫', na=False)
df_sewing = df_raw[sewing_mask].copy()

if '款号' in df_sewing.columns:
    mask = df_sewing['款号'].astype(str).str.startswith(item, na=False)
    df_filtered = df_sewing[mask].copy()
    print(f"\n✅ 篩選完成: {len(df_filtered)} 筆資料")
    print(f"   - 款號: {df_filtered['款号'].unique().tolist()}")
else:
    df_filtered = df_sewing.copy()
    print(f"\n⚠️ 無款號欄位，僅篩選車縫部門")

if len(df_filtered) == 0:
    print(f"\n❌ 沒有符合條件的資料！")
    exit()

# 每日彙總
daily_total = df_filtered.groupby('开单日期')['交货数'].sum()
print(f"\n📊 歷史資料統計:")
print(f"   - 有交貨天數: {len(daily_total)} 天")
print(f"   - 總交貨量: {daily_total.sum():,.0f} 件")
print(f"   - 日均產能: {daily_total.mean():.0f} 件/天")

# ============================================
# 3. 定義多種預測模型
# ============================================

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
        'model_type': '移动平均'
    }

def weighted_moving_average(data, window, forecast_days=30):
    """加權移動平均（近期權重更高）"""
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    
    forecast = []
    # 使用最後window個數據點進行預測
    last_n = data.iloc[-window:].values if len(data) >= window else data.values
    if len(last_n) >= window:
        wma = np.sum(last_n * weights[-len(last_n):])
    else:
        # 數據不足時使用簡單平均
        wma = data.mean()
    
    forecast = [wma] * forecast_days
    
    # 計算歷史WMA
    wma_series = pd.Series(index=data.index, dtype=float)
    for i in range(window-1, len(data)):
        wma_series.iloc[i] = np.sum(data.iloc[i-window+1:i+1].values * weights)
    
    return {
        'name': f'WMA-{window}',
        'forecast': forecast,
        'params': {'window': window, 'weights': 'linear'},
        'last_value': wma,
        'model_type': '加权移动平均',
        'wma_series': wma_series
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
        'model_type': '指数平滑',
        'smooth_series': pd.Series(result, index=data.index)
    }

def double_exponential_smoothing(data, alpha, beta, forecast_days=30):
    """Holt's 線性指數平滑（雙參數）"""
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
        'model_type': '双指数平滑',
        'level_series': pd.Series(level, index=data.index)
    }

def seasonal_naive(data, period=7, forecast_days=30):
    """季節性朴素預測（基於週期）"""
    if len(data) < period:
        return None
    
    forecast = []
    for i in range(forecast_days):
        # 使用上一個週期對應位置的值
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
    print(data)
    """評估模型（使用最後test_days天進行驗證）"""
    if len(data) < test_days + 5:
        return None
    
    train_data = data.iloc[:-test_days]
    test_data = data.iloc[-test_days:]
    
    try:
        # 訓練模型並預測
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
        
        # 計算誤差指標
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
print("🤖 初始化預測模型")
print("=" * 80)

models_config = [
    # 移動平均模型
    {'func': simple_moving_average, 'params': {'window': 3}, 'name': 'SMA-3'},
    {'func': simple_moving_average, 'params': {'window': 7}, 'name': 'SMA-7'},
    {'func': simple_moving_average, 'params': {'window': 15}, 'name': 'SMA-15'},
    {'func': simple_moving_average, 'params': {'window': 30}, 'name': 'SMA-30'},
    
    # 加權移動平均
    {'func': weighted_moving_average, 'params': {'window': 7}, 'name': 'WMA-7'},
    {'func': weighted_moving_average, 'params': {'window': 15}, 'name': 'WMA-15'},
    
    # 指數平滑
    {'func': exponential_smoothing, 'params': {'alpha': 0.3}, 'name': 'ES-α=0.3'},
    {'func': exponential_smoothing, 'params': {'alpha': 0.5}, 'name': 'ES-α=0.5'},
    {'func': exponential_smoothing, 'params': {'alpha': 0.7}, 'name': 'ES-α=0.7'},
    
    # 雙指數平滑
    {'func': double_exponential_smoothing, 'params': {'alpha': 0.5, 'beta': 0.3}, 'name': 'Holt-0.5,0.3'},
    {'func': double_exponential_smoothing, 'params': {'alpha': 0.6, 'beta': 0.2}, 'name': 'Holt-0.6,0.2'},
    
    # 季節性預測
    {'func': seasonal_naive, 'params': {'period': 7}, 'name': 'Seasonal-7'},
    {'func': seasonal_naive, 'params': {'period': 15}, 'name': 'Seasonal-15'},
    {'func': seasonal_naive, 'params': {'period': 30}, 'name': 'Seasonal-30'},
]



# 訓練所有模型
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
            # 評估模型
            evaluation = evaluate_model(daily_total, config['func'], test_days=21, **config['params'])
            # evaluation = evaluate_model(daily_total, config['func'], test_days=, **config['params'])
            
            result['evaluation'] = evaluation
            result['config'] = config
            all_models.append(result)
            print(f"   ✅ {result['name']} - 訓練完成")
    except Exception as e:
        print(f"   ⚠️ {config['name']} 訓練失敗: {e}")

print(f"\n✅ 成功訓練 {len(all_models)} 個模型")

# ============================================
# 5. 模型對比分析
# ============================================
print("\n" + "=" * 80)
print("📊 模型對比分析")
print("=" * 80)

# 比較表格
comparison_data = []
for model in all_models:
    pred_total = sum(model['forecast'])
    pred_daily = pred_total / forecast_days
    hist_daily = daily_total.mean()
    change_pct = ((pred_daily - hist_daily) / hist_daily * 100)
    
    row = {
        '模型名稱': model['name'],
        '模型類型': model['model_type'],
        '參數': str(model['params']),
        '歷史日均(件)': round(hist_daily, 0),
        '預測日均(件)': round(pred_daily, 0),
        '預測30天總產能(件)': round(pred_total, 0),
        '產能變化率(%)': round(change_pct, 1),
    }
    
    if model.get('evaluation'):
        row['驗證MAE'] = round(model['evaluation']['mae'], 0)
        row['驗證RMSE'] = round(model['evaluation']['rmse'], 0)
        row['驗證MAPE(%)'] = round(model['evaluation']['mape'], 1)
    
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('驗證MAE' if '驗證MAE' in comparison_df.columns else '預測日均(件)')

print("\n🏆 模型排名（按預測準確度）:")
print(comparison_df[['模型名稱', '模型類型', '驗證MAE', '驗證MAPE(%)', '預測日均(件)']].head(10).to_string(index=False))

# ============================================
# 6. 生成多模型對比圖表
# ============================================
print("\n" + "=" * 80)
print("📈 生成多模型對比圖表")
print("=" * 80)

# 圖表1: 所有模型預測對比
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'車縫組 {item}* - 多模型預測對比分析', fontsize=16, fontweight='bold')

# 選取最佳模型進行詳細展示
best_model = min(all_models, key=lambda x: x.get('evaluation', {}).get('mae', float('inf')))
worst_model = max(all_models, key=lambda x: x.get('evaluation', {}).get('mae', 0))

# 1. 歷史數據 + 最佳模型預測
ax = axes[0, 0]
future_dates = pd.date_range(start=daily_total.index.max() + timedelta(days=1), periods=forecast_days, freq='D')
ax.plot(daily_total.index[-60:], daily_total.values[-60:], 'b-o', linewidth=1.5, markersize=3, label='歷史實際交貨')
ax.plot(future_dates, best_model['forecast'], 'r--o', linewidth=2, markersize=4, label=f'最佳模型: {best_model["name"]}')
ax.axhline(y=daily_total.mean(), color='blue', linestyle=':', alpha=0.5, label=f'歷史日均: {daily_total.mean():.0f}')
ax.axhline(y=sum(best_model['forecast'])/forecast_days, color='red', linestyle=':', alpha=0.5, label=f'預測日均: {sum(best_model["forecast"])/forecast_days:.0f}')
ax.set_title(f'最佳模型預測 - {best_model["name"]}\nMAE: {best_model.get("evaluation", {}).get("mae", "N/A"):.0f} | MAPE: {best_model.get("evaluation", {}).get("mape", "N/A"):.1f}%', fontsize=11, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('交貨數 (件)')
ax.legend(loc='upper left', fontsize=8)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# 2. 模型誤差對比（柱狀圖）
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
    ax.plot(future_dates, model['forecast'], color=colors[i], linewidth=2, 
            label=f"{model['name']} (MAE:{model.get('evaluation',{}).get('mae',0):.0f})")
ax.plot(daily_total.index[-30:], daily_total.values[-30:], 'b-o', linewidth=1.5, markersize=3, label='歷史數據')
ax.set_title('最佳3個模型預測對比', fontsize=11, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('交貨數 (件)')
ax.legend(loc='upper left', fontsize=8)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# 5. 模型類型性能對比
ax = axes[1, 1]
model_types = {}
for m in all_models:
    if m.get('evaluation'):
        mtype = m['model_type']
        if mtype not in model_types:
            model_types[mtype] = []
        model_types[mtype].append(m.get('evaluation', {}).get('mae', 0))

type_names = list(model_types.keys())
type_mae_means = [np.mean(model_types[t]) for t in type_names]
type_mae_stds = [np.std(model_types[t]) for t in type_names]

bars = ax.bar(type_names, type_mae_means, yerr=type_mae_stds, capsize=5, color='lightgreen', edgecolor='darkgreen')
ax.set_title('各類型模型平均性能對比', fontsize=11, fontweight='bold')
ax.set_ylabel('平均MAE (件)')
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, type_mae_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', ha='center', fontsize=9)

# 6. 預測穩定性分析
ax = axes[1, 2]
stability = []
for m in all_models:
    pred_var = np.var(m['forecast'])
    stability.append({
        'name': m['name'],
        'variance': pred_var,
        'mean': np.mean(m['forecast'])
    })
stability_df = pd.DataFrame(stability).sort_values('variance')
top_stable = stability_df.head(8)

ax.barh(top_stable['name'][:8], top_stable['variance'][:8], color='lightcoral', edgecolor='darkred')
ax.set_title('模型穩定性分析 (預測變異數 - 越低越穩定)', fontsize=11, fontweight='bold')
ax.set_xlabel('預測變異數')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '00_多模型预测对比总图.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   ✅ 多模型對比總圖已儲存")

# ============================================
# 7. 生成詳細的預測趨勢圖
# ============================================
print("\n📊 生成各模型詳細預測圖...")

for i, model in enumerate(all_models[:6]):  # 只顯示前6個模型
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖：歷史數據 + 預測
    ax = axes[0]
    future_dates = pd.date_range(start=daily_total.index.max() + timedelta(days=1), periods=forecast_days, freq='D')
    
    # 顯示最近60天數據
    show_days = min(60, len(daily_total))
    ax.plot(daily_total.index[-show_days:], daily_total.values[-show_days:], 
            'b-o', linewidth=1.5, markersize=3, label='歷史交貨量')
    ax.plot(future_dates, model['forecast'], 'r--o', linewidth=2, markersize=4, label=f'預測值')
    
    # 置信區間（基於歷史標準差）
    hist_std = daily_total.std()
    upper = [model['forecast'][j] + hist_std for j in range(forecast_days)]
    lower = [max(0, model['forecast'][j] - hist_std) for j in range(forecast_days)]
    ax.fill_between(future_dates, lower, upper, alpha=0.2, color='red', label='±1σ 置信區間')
    
    # 添加平均值線
    ax.axhline(y=daily_total.mean(), color='blue', linestyle=':', alpha=0.5, label=f'歷史日均: {daily_total.mean():.0f}')
    ax.axhline(y=sum(model['forecast'])/forecast_days, color='red', linestyle=':', alpha=0.5, 
               label=f'預測日均: {sum(model["forecast"])/forecast_days:.0f}')
    
    ax.set_title(f'{model["name"]} - {model["model_type"]}\n預測30天總產能: {sum(model["forecast"]):,.0f}件', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('交貨數 (件)')
    ax.legend(loc='upper left', fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 右圖：誤差分析（如果有評估數據）
    ax = axes[1]
    if model.get('evaluation'):
        ev = model['evaluation']
        days = list(range(1, len(ev['predictions']) + 1))
        ax.plot(days, ev['actual'], 'b-o', linewidth=2, markersize=6, label='實際值')
        ax.plot(days, ev['predictions'], 'r--s', linewidth=2, markersize=6, label='預測值')
        
        # 添加誤差條
        errors = np.abs(ev['actual'] - ev['predictions'])
        for d, a, p, e in zip(days, ev['actual'], ev['predictions'], errors):
            ax.vlines(d, min(a, p), max(a, p), color='gray', alpha=0.5, linewidth=1)
        
        ax.set_title(f'驗證集表現 (最近7天)\nMAE: {ev["mae"]:.0f} | RMSE: {ev["rmse"]:.0f} | MAPE: {ev["mape"]:.1f}%', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('驗證天數')
        ax.set_ylabel('交貨數 (件)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # 顯示預測值分布
        ax.hist(model['forecast'], bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.axvline(x=np.mean(model['forecast']), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(model["forecast"]):.0f}')
        ax.set_title(f'預測值分布 (30天)', fontsize=12, fontweight='bold')
        ax.set_xlabel('預測交貨數 (件)')
        ax.set_ylabel('頻次')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = model['name'].replace('=', '').replace(',', '_').replace('-', '_')
    plt.savefig(os.path.join(output_folder, f'{safe_name}_详细预测.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

print(f"   ✅ 已生成 {min(6, len(all_models))} 個詳細預測圖")

# ============================================
# 8. 生成Excel報告
# ============================================
print("\n📊 生成Excel綜合報告...")

excel_file = os.path.join(output_folder, '00_多模型预测综合报告.xlsx')

try:
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # 模型對比總表
        comparison_df.to_excel(writer, sheet_name='模型對比總表', index=False)
        
        # 最佳模型詳細預測
        best_model_forecast = pd.DataFrame({
            '預測日期': future_dates,
            '預測交貨數': [round(v) for v in best_model['forecast']],
            '下限(±1σ)': [max(0, round(v - daily_total.std())) for v in best_model['forecast']],
            '上限(±1σ)': [round(v + daily_total.std()) for v in best_model['forecast']]
        })
        best_model_forecast.to_excel(writer, sheet_name=f'最佳模型_{best_model["name"]}_預測', index=False)
        
        # 所有模型預測匯總
        all_forecasts = pd.DataFrame({'預測日期': future_dates})
        for model in all_models:
            all_forecasts[model['name']] = [round(v) for v in model['forecast']]
        all_forecasts.to_excel(writer, sheet_name='所有模型預測匯總', index=False)
        
        # 統計摘要
        summary = pd.DataFrame({
            '統計項目': [
                '歷史總交貨量(件)',
                '歷史有交貨天數',
                '歷史日均產能(件/天)',
                '歷史標準差(件)',
                '評估模型數量',
                '最佳模型(按MAE)',
                '最佳模型MAE(件)',
                '最佳模型MAPE(%)',
                '最穩定模型(按變異數)',
                '多模型平均預測日均(件/天)',
                '多模型預測範圍(件/天)'
            ],
            '數值': [
                f"{daily_total.sum():,.0f}",
                len(daily_total),
                f"{daily_total.mean():.0f}",
                f"{daily_total.std():.0f}",
                len(all_models),
                best_model['name'],
                f"{best_model.get('evaluation', {}).get('mae', 'N/A')}",
                f"{best_model.get('evaluation', {}).get('mape', 'N/A'):.1f}%" if best_model.get('evaluation') else 'N/A',
                stability_df.iloc[0]['name'] if len(stability_df) > 0 else 'N/A',
                f"{pred_mean.mean():.0f}",
                f"{pred_min.mean():.0f} ~ {pred_max.mean():.0f}"
            ]
        })
        summary.to_excel(writer, sheet_name='統計摘要', index=False)
        
        # 模型推薦
        recommendation = pd.DataFrame({
            '使用場景': ['短期波動大', '一般生產規劃', '平穩生產', '保守預測', '樂觀預測'],
            '推薦模型': [
                'SMA-3 或 ES-α=0.7',
                'SMA-7 或 WMA-7',
                'SMA-15 或 ES-α=0.3',
                f'{min(all_models, key=lambda x: sum(x["forecast"]))["name"]}',
                f'{max(all_models, key=lambda x: sum(x["forecast"]))["name"]}'
            ],
            '預測日均(件)': [
                next((m['forecast'][0] for m in all_models if m['name'] == 'SMA-3'), 'N/A'),
                next((sum(m['forecast'])/forecast_days for m in all_models if m['name'] == 'SMA-7'), 'N/A'),
                next((sum(m['forecast'])/forecast_days for m in all_models if m['name'] == 'SMA-15'), 'N/A'),
                f"{min(all_models, key=lambda x: sum(x['forecast']))['forecast'][0]:.0f}",
                f"{max(all_models, key=lambda x: sum(x['forecast']))['forecast'][0]:.0f}"
            ]
        })
        recommendation.to_excel(writer, sheet_name='模型推薦', index=False)
    
    print(f"   ✅ Excel報告已儲存: {excel_file}")
except Exception as e:
    print(f"   ⚠️ Excel儲存失敗: {e}")

# ============================================
# 9. 最終報告輸出
# ============================================
print("\n" + "=" * 80)
print("📋 多模型預測最終報告")
print("=" * 80)

print(f"\n🏆 最佳模型: {best_model['name']}")
if best_model.get('evaluation'):
    print(f"   - MAE (平均絕對誤差): {best_model['evaluation']['mae']:.0f} 件")
    print(f"   - RMSE (均方根誤差): {best_model['evaluation']['rmse']:.0f} 件")
    print(f"   - MAPE (平均絕對百分比誤差): {best_model['evaluation']['mape']:.1f}%")
print(f"   - 預測30天總產能: {sum(best_model['forecast']):,.0f} 件")
print(f"   - 預測日均產能: {sum(best_model['forecast'])/forecast_days:.0f} 件/天")

print(f"\n📊 多模型統計:")
print(f"   - 模型總數: {len(all_models)}")
print(f"   - 預測日均範圍: {pred_min.mean():.0f} ~ {pred_max.mean():.0f} 件/天")
print(f"   - 預測30天總產能範圍: {pred_min.sum():.0f} ~ {pred_max.sum():.0f} 件")

print(f"\n📁 輸出檔案結構:")
print(f"\n{output_folder}/")
print("   ├── 00_多模型预测综合报告.xlsx          (所有模型對比)")
print("   ├── 00_多模型预测对比总图.png          (6合1對比圖)")
print("   ├── [模型名稱]_详细预测.png            (各模型詳細圖)")
print("   └── ...")

print("\n" + "=" * 80)
print("⚠️ 重要說明:")
print(f"   1. 本報告使用{test_days}天歷史數據進行模型驗證")
print("   2. MAE/越低代表模型越準確")
print("   3. 多模型平均可降低單一模型風險")
print("   4. 建議根據實際生產情況選擇合適模型")
print("   5. 定期更新數據可提高預測準確度")
print("=" * 80)

print("\n✨ 多模型預測對比報告生成完成！")
print(f"📁 報告已儲存至: {os.path.abspath(output_folder)}")
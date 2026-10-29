import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
from math import sqrt
warnings.filterwarnings('ignore')


class SewingForecastAnalyzer:
    """
    車縫組生產預測分析器
    支持多模型預測對比，自動排除星期日
    """
    
    def __init__(self, file_path, items, test_days=45, forecast_days=30, output_folder=None):
        """
        初始化預測分析器
        
        Parameters:
        -----------
        file_path : str
            Excel文件路徑
        items : list
            款號列表
        test_days : int
            驗證集天數
        forecast_days : int
            預測天數
        output_folder : str
            輸出資料夾名稱，默認為自動生成
        """
        self.file_path = file_path
        self.items = items
        self.test_days = test_days
        self.forecast_days = forecast_days
        
        # 設置輸出資料夾
        if output_folder is None:
            self.output_folder = f'车缝组_多模型预测对比报告(排除周日)'
        else:
            self.output_folder = output_folder
        
        # 數據屬性
        self.df_raw = None
        self.df_filtered = None
        self.daily_total_all = None
        self.daily_total = None
        self.all_forecast_dates = None
        self.all_models = []
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建輸出資料夾
        self._create_output_folder()
    
    def _create_output_folder(self):
        """創建輸出資料夾"""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"📁 創建資料夾: {self.output_folder}")
    
    def load_data(self):
        """讀取並預處理Excel資料"""
        print("=" * 80)
        print(f"📊 車縫組 - 款號* 多模型預測對比報告 (排除星期日)")
        print("=" * 80)
        print(f"\n📂 讀取檔案: {self.file_path}")
        
        try:
            self.df_raw = pd.read_excel(self.file_path, sheet_name='生产每日交货清单')
            print(f"✅ 成功讀取 {len(self.df_raw)} 筆資料")
        except FileNotFoundError:
            print(f"❌ 檔案不存在: {self.file_path}")
            raise
        except Exception as e:
            print(f"❌ 讀取失敗: {e}")
            raise
        
        # 檢查必要欄位
        required_columns = ['开单日期', '发出部门', '交货数']
        for col in required_columns:
            if col not in self.df_raw.columns:
                raise ValueError(f"缺少必要欄位: {col}")
        
        # 資料清理
        self.df_raw['开单日期'] = pd.to_datetime(self.df_raw['开单日期'])
        self.df_raw = self.df_raw.dropna(subset=['开单日期', '发出部门', '交货数'])
        self.df_raw = self.df_raw[self.df_raw['交货数'] > 0]
        self.df_raw = self.df_raw.sort_values('开单日期')
        
        return self
    
    def filter_data(self):
        """篩選車縫部門和指定款號"""
        print("\n" + "=" * 80)
        print("🔍 篩選條件:")
        print("   - 發出部門 包含「車縫」")
        print(f"   - 款號 以下款號：{self.items}")
        print("   - 交貨數 > 0")
        print("   - 排除星期日（星期天不預測）")
        print("=" * 80)
        
        sewing_mask = self.df_raw['发出部门'].str.contains('車縫', na=False)
        df_sewing = self.df_raw[sewing_mask].copy()
        
        if '款号' in df_sewing.columns:
            mask = df_sewing['款号'].isin(self.items)
            self.df_filtered = df_sewing[mask].copy()
            print(f"\n✅ 篩選完成: {len(self.df_filtered)} 筆資料")
            print(f"   - 款號: {self.df_filtered['款号'].unique().tolist()}")
        else:
            self.df_filtered = df_sewing.copy()
            print(f"\n⚠️ 無款號欄位，僅篩選車縫部門")
        
        if len(self.df_filtered) == 0:
            raise ValueError("沒有符合條件的資料！")
        
        # 每日彙總（包含所有日期）
        self.daily_total_all = self.df_filtered.groupby('开单日期')['交货数'].sum()
        
        # 過濾掉星期日的數據（用於模型訓練）
        self.df_filtered['星期幾'] = self.df_filtered['开单日期'].dt.dayofweek
        df_filtered_no_sunday = self.df_filtered[self.df_filtered['星期幾'] != 6].copy()
        self.daily_total = df_filtered_no_sunday.groupby('开单日期')['交货数'].sum()
        
        print(f"\n📊 歷史資料統計:")
        print(f"   - 原始有交貨天數: {len(self.daily_total_all)} 天")
        print(f"   - 排除星期日後天數: {len(self.daily_total)} 天")
        print(f"   - 排除星期日數: {len(self.daily_total_all) - len(self.daily_total)} 天")
        print(f"   - 總交貨量: {self.daily_total.sum():,.0f} 件")
        print(f"   - 日均產能(僅工作日): {self.daily_total.mean():.0f} 件/天")
        
        return self
    
    def get_all_forecast_dates(self):
        """獲取包含星期日的完整預測日期列表"""
        last_date = self.daily_total.index.max()
        start_forecast_date = last_date + timedelta(days=1)
        self.all_forecast_dates = [start_forecast_date + timedelta(days=i) for i in range(self.forecast_days)]
        return self.all_forecast_dates
    
    # ========== 預測模型方法 ==========
    
    def simple_moving_average(self, data, window, forecast_days=None):
        """簡單移動平均"""
        if forecast_days is None:
            forecast_days = self.forecast_days
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
    
    def weighted_moving_average(self, data, window, forecast_days=None):
        """加權移動平均"""
        if forecast_days is None:
            forecast_days = self.forecast_days
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
    
    def exponential_smoothing(self, data, alpha, forecast_days=None):
        """指數平滑"""
        if forecast_days is None:
            forecast_days = self.forecast_days
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
    
    def double_exponential_smoothing(self, data, alpha, beta, forecast_days=None):
        """Holt's 線性指數平滑"""
        if forecast_days is None:
            forecast_days = self.forecast_days
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
    
    def seasonal_naive(self, data, period=7, forecast_days=None):
        """季節性朴素預測"""
        if forecast_days is None:
            forecast_days = self.forecast_days
        
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
    
    def evaluate_model(self, model_func, **kwargs):
        """評估模型在驗證集上的表現"""
        data = self.daily_total
        
        if len(data) < self.test_days + 5:
            return None
        
        train_data = data.iloc[:-self.test_days]
        test_data = data.iloc[-self.test_days:]
        
        try:
            result = model_func(train_data, forecast_days=self.test_days, **kwargs)
            if result is None:
                return None
            
            predictions = result['forecast'][:self.test_days]
            
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
    
    def train_models(self):
        """訓練所有定義的模型"""
        print("\n" + "=" * 80)
        print("🤖 初始化預測模型 (排除星期日)")
        print("=" * 80)
        
        models_config = [
            {'func': self.simple_moving_average, 'params': {'window': 3}, 'name': 'SMA-3'},
            {'func': self.simple_moving_average, 'params': {'window': 7}, 'name': 'SMA-7'},
            {'func': self.simple_moving_average, 'params': {'window': 15}, 'name': 'SMA-15'},
            {'func': self.simple_moving_average, 'params': {'window': 30}, 'name': 'SMA-30'},
            {'func': self.weighted_moving_average, 'params': {'window': 7}, 'name': 'WMA-7'},
            {'func': self.weighted_moving_average, 'params': {'window': 15}, 'name': 'WMA-15'},
            {'func': self.exponential_smoothing, 'params': {'alpha': 0.3}, 'name': 'ES-α=0.3'},
            {'func': self.exponential_smoothing, 'params': {'alpha': 0.5}, 'name': 'ES-α=0.5'},
            {'func': self.exponential_smoothing, 'params': {'alpha': 0.7}, 'name': 'ES-α=0.7'},
            {'func': self.double_exponential_smoothing, 'params': {'alpha': 0.5, 'beta': 0.3}, 'name': 'Holt-0.5,0.3'},
            {'func': self.double_exponential_smoothing, 'params': {'alpha': 0.6, 'beta': 0.2}, 'name': 'Holt-0.6,0.2'},
            {'func': self.seasonal_naive, 'params': {'period': 7}, 'name': 'Seasonal-7'},
            {'func': self.seasonal_naive, 'params': {'period': 15}, 'name': 'Seasonal-15'},
            {'func': self.seasonal_naive, 'params': {'period': 30}, 'name': 'Seasonal-30'},
        ]
        
        self.all_models = []
        print("\n📊 訓練模型中...")
        
        for config in models_config:
            try:
                result = config['func'](self.daily_total, forecast_days=self.forecast_days, **config['params'])
                
                if result is not None:
                    evaluation = self.evaluate_model(config['func'], **config['params'])
                    result['evaluation'] = evaluation
                    result['config'] = config
                    self.all_models.append(result)
                    print(f"   ✅ {result['name']} - 訓練完成")
            except Exception as e:
                print(f"   ⚠️ {config['name']} 訓練失敗: {e}")
        
        print(f"\n✅ 成功訓練 {len(self.all_models)} 個模型")
        return self.all_models
    
    def apply_sunday_filter(self):
        """對預測結果應用星期日過濾"""
        print("\n" + "=" * 80)
        print("📅 應用星期日過濾（星期日預測值設為0）")
        print("=" * 80)
        
        # 計算星期日數量
        sunday_count = sum(1 for d in self.all_forecast_dates if d.weekday() == 6)
        workday_count = self.forecast_days - sunday_count
        
        print(f"   - 預測期間: {self.all_forecast_dates[0].strftime('%Y-%m-%d')} ~ {self.all_forecast_dates[-1].strftime('%Y-%m-%d')}")
        print(f"   - 星期日天數: {sunday_count} 天 (預測值設為0)")
        print(f"   - 工作日天數: {workday_count} 天")
        
        for model in self.all_models:
            filtered_forecast = []
            workday_idx = 0
            for forecast_date in self.all_forecast_dates:
                if forecast_date.weekday() == 6:  # 星期日
                    filtered_forecast.append(0)
                else:
                    if workday_idx < len(model['forecast']):
                        filtered_forecast.append(model['forecast'][workday_idx])
                        workday_idx += 1
                    else:
                        filtered_forecast.append(0)
            
            model['original_forecast'] = model['forecast']
            model['forecast'] = filtered_forecast
        
        print(f"   ✅ 已對 {len(self.all_models)} 個模型應用星期日過濾")
    
    def get_comparison_dataframe(self):
        """生成模型對比數據表"""
        comparison_data = []
        for model in self.all_models:
            pred_total = sum(model['forecast'])
            pred_daily = pred_total / self.forecast_days
            hist_daily = self.daily_total.mean()
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
        
        return comparison_df
    
    def plot_comparison_charts(self):
        """生成多模型對比圖表"""
        print("\n" + "=" * 80)
        print("📈 生成多模型對比圖表")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'車縫組* - 多模型預測對比分析 (排除星期日)', fontsize=16, fontweight='bold')
        
        best_model = min(self.all_models, key=lambda x: x.get('evaluation', {}).get('mae', float('inf')))
        
        # 1. 歷史數據 + 最佳模型預測
        ax = axes[0, 0]
        ax.plot(self.daily_total.index[-60:], self.daily_total.values[-60:], 
                'b-o', linewidth=1.5, markersize=3, label='歷史實際交貨')
        
        non_sunday_dates = [self.all_forecast_dates[i] for i in range(self.forecast_days) 
                           if self.all_forecast_dates[i].weekday() != 6]
        non_sunday_values = [best_model['forecast'][i] for i in range(self.forecast_days) 
                            if self.all_forecast_dates[i].weekday() != 6]
        sunday_dates = [self.all_forecast_dates[i] for i in range(self.forecast_days) 
                       if self.all_forecast_dates[i].weekday() == 6]
        
        ax.plot(non_sunday_dates, non_sunday_values, 'r--o', linewidth=2, markersize=4, 
                label=f'最佳模型: {best_model["name"]}')
        if sunday_dates:
            ax.scatter(sunday_dates, [0]*len(sunday_dates), color='gray', marker='x', s=50, 
                      label='星期日(不生產)')
        
        ax.axhline(y=self.daily_total.mean(), color='blue', linestyle=':', alpha=0.5, 
                   label=f'歷史日均: {self.daily_total.mean():.0f}')
        ax.axhline(y=sum(best_model['forecast'])/self.forecast_days, color='red', linestyle=':', alpha=0.5, 
                   label=f'預測日均(含周日): {sum(best_model["forecast"])/self.forecast_days:.0f}')
        ax.set_title(f'最佳模型預測 - {best_model["name"]}\nMAE: {best_model.get("evaluation", {}).get("mae", "N/A"):.0f} | MAPE: {best_model.get("evaluation", {}).get("mape", "N/A"):.1f}%', 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('交貨數 (件)')
        ax.legend(loc='upper left', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 2. 模型誤差對比
        ax = axes[0, 1]
        models_with_mae = [(m['name'], m.get('evaluation', {}).get('mae', 0)) 
                          for m in self.all_models if m.get('evaluation')]
        models_with_mae.sort(key=lambda x: x[1])
        if models_with_mae:
            names = [m[0][:15] for m in models_with_mae[:10]]
            maes = [m[1] for m in models_with_mae[:10]]
            bars = ax.barh(names, maes, color='skyblue', edgecolor='navy')
            ax.set_title('模型誤差對比 (MAE - 越低越好)', fontsize=11, fontweight='bold')
            ax.set_xlabel('平均絕對誤差 (件)')
            ax.grid(True, alpha=0.3, axis='x')
            for bar, val in zip(bars, maes):
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f'{val:.0f}', 
                       va='center', fontsize=9)
        
        # 3. 模型預測範圍對比
        ax = axes[0, 2]
        all_predictions = [m['forecast'] for m in self.all_models]
        pred_array = np.array(all_predictions)
        pred_mean = pred_array.mean(axis=0)
        pred_std = pred_array.std(axis=0)
        pred_min = pred_array.min(axis=0)
        pred_max = pred_array.max(axis=0)
        
        ax.fill_between(range(self.forecast_days), pred_min, pred_max, alpha=0.3, color='gray', 
                       label='預測範圍(最小-最大)')
        ax.plot(range(self.forecast_days), pred_mean, 'b-', linewidth=2, label='平均預測')
        ax.plot(range(self.forecast_days), [self.daily_total.mean()] * self.forecast_days, 'r--', 
               label='歷史日均線')
        ax.set_title(f'多模型預測範圍 (共{len(self.all_models)}個模型, 排除星期日)', fontsize=11, fontweight='bold')
        ax.set_xlabel('預測天數')
        ax.set_ylabel('預測交貨數 (件)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 最佳3個模型預測對比
        ax = axes[1, 0]
        top_3 = sorted(self.all_models, key=lambda x: x.get('evaluation', {}).get('mae', float('inf')))[:3]
        colors = ['red', 'green', 'orange']
        for i, model in enumerate(top_3):
            non_sunday_vals = [model['forecast'][j] for j in range(self.forecast_days) 
                              if self.all_forecast_dates[j].weekday() != 6]
            ax.plot(non_sunday_dates, non_sunday_vals, color=colors[i], linewidth=2, 
                    label=f"{model['name']} (MAE:{model.get('evaluation',{}).get('mae',0):.0f})")
        ax.plot(self.daily_total.index[-30:], self.daily_total.values[-30:], 'b-o', linewidth=1.5, markersize=3, 
               label='歷史數據')
        ax.set_title('最佳3個模型預測對比', fontsize=11, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('交貨數 (件)')
        ax.legend(loc='upper left', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 5. 模型類型性能對比
        ax = axes[1, 1]
        model_types = {}
        for m in self.all_models:
            if m.get('evaluation'):
                mtype = m['model_type']
                if mtype not in model_types:
                    model_types[mtype] = []
                model_types[mtype].append(m.get('evaluation', {}).get('mae', 0))
        
        type_names = list(model_types.keys())
        type_mae_means = [np.mean(model_types[t]) for t in type_names]
        type_mae_stds = [np.std(model_types[t]) for t in type_names]
        
        bars = ax.bar(type_names, type_mae_means, yerr=type_mae_stds, capsize=5, 
                     color='lightgreen', edgecolor='darkgreen')
        ax.set_title('各類型模型平均性能對比', fontsize=11, fontweight='bold')
        ax.set_ylabel('平均MAE (件)')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, type_mae_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', 
                   ha='center', fontsize=9)
        
        # 6. 星期日影響分析
        ax = axes[1, 2]
        daily_total_all_with_weekday = self.df_filtered.copy()
        daily_total_all_with_weekday['星期'] = daily_total_all_with_weekday['开单日期'].dt.dayofweek
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekday_avg = daily_total_all_with_weekday.groupby('星期')['交货数'].mean()
        weekday_avg.index = weekday_names
        
        colors_bar = ['steelblue'] * 6 + ['lightcoral']
        bars = ax.bar(weekday_names, weekday_avg.values, color=colors_bar, edgecolor='black')
        ax.set_title('各星期平均交貨量分析', fontsize=11, fontweight='bold')
        ax.set_xlabel('星期')
        ax.set_ylabel('平均交貨數 (件)')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, weekday_avg.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', 
                   ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '00_多模型预测对比总图.png'), dpi=150, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ 多模型對比總圖已儲存")
        
        return best_model
    
    def plot_detailed_forecasts(self, max_models=6):
        """生成各模型詳細預測圖"""
        print("\n📊 生成各模型詳細預測圖...")
        
        for i, model in enumerate(self.all_models[:max_models]):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            ax = axes[0]
            show_days = min(60, len(self.daily_total))
            ax.plot(self.daily_total.index[-show_days:], self.daily_total.values[-show_days:], 
                    'b-o', linewidth=1.5, markersize=3, label='歷史交貨量')
            
            non_sunday_dates = [self.all_forecast_dates[j] for j in range(self.forecast_days) 
                               if self.all_forecast_dates[j].weekday() != 6]
            non_sunday_vals = [model['forecast'][j] for j in range(self.forecast_days) 
                              if self.all_forecast_dates[j].weekday() != 6]
            sunday_dates = [self.all_forecast_dates[j] for j in range(self.forecast_days) 
                           if self.all_forecast_dates[j].weekday() == 6]
            
            if non_sunday_dates:
                ax.plot(non_sunday_dates, non_sunday_vals, 'r--o', linewidth=2, markersize=4, label='預測值')
            if sunday_dates:
                ax.scatter(sunday_dates, [0]*len(sunday_dates), color='gray', marker='x', s=50, 
                          label='星期日(不生產)')
            
            hist_std = self.daily_total.std()
            upper = [model['forecast'][j] + hist_std for j in range(self.forecast_days)]
            lower = [max(0, model['forecast'][j] - hist_std) for j in range(self.forecast_days)]
            ax.fill_between(self.all_forecast_dates, lower, upper, alpha=0.2, color='red', label='±1σ 置信區間')
            
            ax.axhline(y=self.daily_total.mean(), color='blue', linestyle=':', alpha=0.5, 
                      label=f'歷史日均: {self.daily_total.mean():.0f}')
            ax.axhline(y=sum(model['forecast'])/self.forecast_days, color='red', linestyle=':', alpha=0.5, 
                      label=f'預測日均(含周日): {sum(model["forecast"])/self.forecast_days:.0f}')
            
            workday_avg = np.mean(non_sunday_vals) if non_sunday_vals else 0
            ax.axhline(y=workday_avg, color='orange', linestyle=':', alpha=0.5, 
                      label=f'預測工作日日均: {workday_avg:.0f}')
            
            ax.set_title(f'{model["name"]} - {model["model_type"]}\n預測30天總產能: {sum(model["forecast"]):,.0f}件 (星期日不生產)', 
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
                
                ax.set_title(f'驗證集表現 (最近{self.test_days}個工作日)\nMAE: {ev["mae"]:.0f} | RMSE: {ev["rmse"]:.0f} | MAPE: {ev["mape"]:.1f}%', 
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
            plt.savefig(os.path.join(self.output_folder, f'{safe_name}_详细预测.png'), dpi=150, 
                       bbox_inches='tight', facecolor='white')
            plt.close()
        
        print(f"   ✅ 已生成 {min(max_models, len(self.all_models))} 個詳細預測圖")
    
    def save_excel_report(self, best_model, comparison_df, pred_mean, pred_min, pred_max):
        """保存Excel綜合報告"""
        print("\n📊 生成Excel綜合報告...")
        
        excel_file = os.path.join(self.output_folder, '00_多模型预测综合报告.xlsx')
        
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 原始資料
                self.df_filtered.to_excel(writer, sheet_name='原始資料', index=False)
                pd.DataFrame(self.items, columns=['款号']).to_excel(writer, sheet_name='觀測款號', index=False)
                comparison_df.to_excel(writer, sheet_name='模型對比總表', index=False)
                
                best_model_forecast = pd.DataFrame({
                    '預測日期': self.all_forecast_dates,
                    '星期': [['周一', '周二', '周三', '周四', '周五', '周六', '周日'][d.weekday()] for d in self.all_forecast_dates],
                    '預測交貨數': [round(v) for v in best_model['forecast']],
                    '下限(±1σ)': [max(0, round(v - self.daily_total.std())) for v in best_model['forecast']],
                    '上限(±1σ)': [round(v + self.daily_total.std()) for v in best_model['forecast']]
                })
                best_model_forecast.to_excel(writer, sheet_name=f'最佳模型_{best_model["name"]}_預測', index=False)
                
                all_forecasts = pd.DataFrame({
                    '預測日期': self.all_forecast_dates,
                    '星期': [['周一', '周二', '周三', '周四', '周五', '周六', '周日'][d.weekday()] for d in self.all_forecast_dates]
                })
                for model in self.all_models:
                    all_forecasts[model['name']] = [round(v) for v in model['forecast']]
                all_forecasts.to_excel(writer, sheet_name='所有模型預測匯總', index=False)
                
                sunday_count = sum(1 for d in self.all_forecast_dates if d.weekday() == 6)
                
                summary = pd.DataFrame({
                    '統計項目': [
                        '歷史總交貨量(件)',
                        '歷史有交貨天數(總)',
                        '歷史有交貨天數(排除周日)',
                        '歷史日均產能(件/天)',
                        '歷史標準差(件)',
                        '評估模型數量',
                        '最佳模型(按MAE)',
                        '最佳模型MAE(件)',
                        '最佳模型MAPE(%)',
                        '多模型平均預測日均(件/天)',
                        '多模型預測範圍(件/天)',
                        '星期日處理方式'
                    ],
                    '數值': [
                        f"{self.daily_total.sum():,.0f}",
                        len(self.daily_total_all),
                        len(self.daily_total),
                        f"{self.daily_total.mean():.0f}",
                        f"{self.daily_total.std():.0f}",
                        len(self.all_models),
                        best_model['name'],
                        f"{best_model.get('evaluation', {}).get('mae', 'N/A')}",
                        f"{best_model.get('evaluation', {}).get('mape', 'N/A'):.1f}%" if best_model.get('evaluation') else 'N/A',
                        f"{pred_mean.mean():.0f}",
                        f"{pred_min.mean():.0f} ~ {pred_max.mean():.0f}",
                        '星期日預測值設為0（不生產）'
                    ]
                })
                summary.to_excel(writer, sheet_name='統計摘要', index=False)
                
                recommendation = pd.DataFrame({
                    '使用場景': ['短期波動大', '一般生產規劃', '平穩生產', '保守預測', '樂觀預測'],
                    '推薦模型': [
                        'SMA-3 或 ES-α=0.7',
                        'SMA-7 或 WMA-7',
                        'SMA-15 或 ES-α=0.3',
                        f'{min(self.all_models, key=lambda x: sum(x["forecast"]))["name"]}',
                        f'{max(self.all_models, key=lambda x: sum(x["forecast"]))["name"]}'
                    ],
                    '預測工作日日均(件)': [
                        'N/A', 'N/A', 'N/A',
                        f"{sum(min(self.all_models, key=lambda x: sum(x['forecast']))['forecast'])/self.forecast_days:.0f}",
                        f"{sum(max(self.all_models, key=lambda x: sum(x['forecast']))['forecast'])/self.forecast_days:.0f}"
                    ]
                })
                recommendation.to_excel(writer, sheet_name='模型推薦', index=False)
            
            print(f"   ✅ Excel報告已儲存: {excel_file}")
        except Exception as e:
            print(f"   ⚠️ Excel儲存失敗: {e}")
    
    def print_final_report(self, best_model, pred_mean, pred_min, pred_max, sunday_count):
        """打印最終報告"""
        print("\n" + "=" * 80)
        print("📋 多模型預測最終報告 (排除星期日)")
        print("=" * 80)
        
        print(f"\n🏆 最佳模型: {best_model['name']}")
        if best_model.get('evaluation'):
            print(f"   - MAE (平均絕對誤差): {best_model['evaluation']['mae']:.0f} 件")
            print(f"   - RMSE (均方根誤差): {best_model['evaluation']['rmse']:.0f} 件")
            print(f"   - MAPE (平均絕對百分比誤差): {best_model['evaluation']['mape']:.1f}%")
        print(f"   - 預測30天總產能: {sum(best_model['forecast']):,.0f} 件")
        print(f"   - 預測日均產能(含周日): {sum(best_model['forecast'])/self.forecast_days:.0f} 件/天")
        workday_forecast = [v for v in best_model['forecast'] if v > 0]
        if workday_forecast:
            print(f"   - 預測工作日日均產能: {np.mean(workday_forecast):.0f} 件/天")
        
        print(f"\n📊 多模型統計:")
        print(f"   - 模型總數: {len(self.all_models)}")
        print(f"   - 預測日均範圍(含周日): {pred_min.mean():.0f} ~ {pred_max.mean():.0f} 件/天")
        print(f"   - 預測30天總產能範圍: {pred_min.sum():.0f} ~ {pred_max.sum():.0f} 件")
        
        print(f"\n📅 星期日處理說明:")
        print(f"   - 歷史數據訓練時已排除星期日")
        print(f"   - 未來預測中星期日統一設為0")
        print(f"   - 共排除 {len(self.daily_total_all) - len(self.daily_total)} 天歷史星期日數據")
        print(f"   - 預測期間共有 {sunday_count} 個星期日")
        
        print(f"\n📁 輸出檔案結構:")
        print(f"\n{self.output_folder}/")
        print("   ├── 00_多模型预测综合报告.xlsx          (所有模型對比)")
        print("   ├── 00_多模型预测对比总图.png          (6合1對比圖)")
        print("   ├── [模型名稱]_详细预测.png            (各模型詳細圖)")
        print("   └── ...")
        
        print("\n" + "=" * 80)
        print("⚠️ 重要說明:")
        print(f"   1. 本報告使用{self.test_days}天歷史數據進行模型驗證")
        print("   2. 星期日不生產，預測值設為0")
        print("   3. 歷史訓練數據已排除星期日")
        print("   4. MAE越低代表模型越準確")
        print("   5. 預測工作日日均更能反映實際產能")
        print("=" * 80)
        
        print("\n✨ 多模型預測對比報告生成完成！")
        print(f"📁 報告已儲存至: {os.path.abspath(self.output_folder)}")
    
    def run(self):
        """執行完整的預測流程"""
        # 1. 加載數據
        self.load_data()
        
        # 2. 篩選數據
        self.filter_data()
        
        # 3. 獲取預測日期
        self.get_all_forecast_dates()
        
        # 4. 訓練所有模型
        self.train_models()
        
        # 5. 應用星期日過濾
        self.apply_sunday_filter()
        
        # 6. 獲取對比數據表
        comparison_df = self.get_comparison_dataframe()
        
        # 7. 生成對比圖表
        best_model = self.plot_comparison_charts()
        
        # 8. 生成詳細預測圖
        self.plot_detailed_forecasts()
        
        # 9. 計算統計數據
        all_predictions = [m['forecast'] for m in self.all_models]
        pred_array = np.array(all_predictions)
        pred_min = pred_array.min(axis=0)
        pred_max = pred_array.max(axis=0)
        pred_mean = pred_array.mean(axis=0)
        
        sunday_count = sum(1 for d in self.all_forecast_dates if d.weekday() == 6)
        
        # 10. 保存Excel報告
        self.save_excel_report(best_model, comparison_df, pred_mean, pred_min, pred_max)
        
        # 11. 打印最終報告
        self.print_final_report(best_model, pred_mean, pred_min, pred_max, sunday_count)
        
        return best_model


# ============================================
# 使用範例
# ============================================
if __name__ == "__main__":
    # 配置參數
    items = ['P12573-3L', 'P12573-3H']
    file_path = r'C:\Users\Dell\Downloads\生产每日交货清单.xlsx'
    
    # 創建分析器實例並運行
    analyzer = SewingForecastAnalyzer(
        file_path=file_path,
        items=items,
        test_days=45,
        forecast_days=30,
        output_folder='车缝组_多模型预测对比报告(排除周日)'
    )
    
    # 執行完整分析
    best_model = analyzer.run()
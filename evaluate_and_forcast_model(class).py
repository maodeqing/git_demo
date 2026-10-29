import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import duckdb as dd

warnings.filterwarnings('ignore')


class SewingForecastAnalyzer:
    """
    車縫組生產預測分析器
    支持多模型預測對比，自動排除星期日
    包含預測與效能評估功能
    新增回歸分析模型（线性回归、多项式回归、岭回归、自回归等）
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
        
        # 評估相關屬性
        self.actual_production_data = None
        self.evaluation_results = None
        self.category_evaluation = None
        
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
    
    # ========== 預測模型方法（已添加負值保護） ==========
    
    def simple_moving_average(self, data, window, forecast_days=None):
        """簡單移動平均（保證非負）"""
        if forecast_days is None:
            forecast_days = self.forecast_days
        ma = data.rolling(window=window, min_periods=1).mean()
        last_ma = max(0, ma.iloc[-1])  # 確保非負
        forecast = [last_ma] * forecast_days
        
        return {
            'name': f'SMA-{window}',
            'forecast': forecast,
            'params': {'window': window},
            'last_value': last_ma,
            'model_type': '移动平均'
        }
    
    def weighted_moving_average(self, data, window, forecast_days=None):
        """加權移動平均（保證非負）"""
        if forecast_days is None:
            forecast_days = self.forecast_days
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        
        last_n = data.iloc[-window:].values if len(data) >= window else data.values
        if len(last_n) >= window:
            wma = np.sum(last_n * weights[-len(last_n):])
        else:
            wma = data.mean()
        
        wma = max(0, wma)  # 確保非負
        forecast = [wma] * forecast_days
        
        return {
            'name': f'WMA-{window}',
            'forecast': forecast,
            'params': {'window': window, 'weights': 'linear'},
            'last_value': wma,
            'model_type': '加权移动平均'
        }
    
    def exponential_smoothing(self, data, alpha, forecast_days=None):
        """指數平滑（保證非負）"""
        if forecast_days is None:
            forecast_days = self.forecast_days
        data_clean = data.clip(lower=0)
        result = [data_clean.iloc[0]]
        for t in range(1, len(data_clean)):
            result.append(alpha * data_clean.iloc[t] + (1 - alpha) * result[-1])
        
        last_smooth = max(0, result[-1])  # 確保非負
        forecast = [last_smooth] * forecast_days
        
        return {
            'name': f'ES-α={alpha}',
            'forecast': forecast,
            'params': {'alpha': alpha},
            'last_value': last_smooth,
            'model_type': '指数平滑'
        }
    
    def double_exponential_smoothing(self, data, alpha, beta, forecast_days=None):
        """Holt's 線性指數平滑（保證預測值不為負）"""
        if forecast_days is None:
            forecast_days = self.forecast_days
        n = len(data)
        
        # 確保數據非負
        data_clean = data.clip(lower=0)
        
        if n < 2:
            # 數據不足時返回簡單平均值
            avg = max(0, data_clean.mean())
            forecast = [avg] * forecast_days
            return {
                'name': f'Holt-α={alpha},β={beta}',
                'forecast': forecast,
                'params': {'alpha': alpha, 'beta': beta},
                'last_value': avg,
                'model_type': '双指数平滑'
            }
        
        level = [data_clean.iloc[0]]
        trend = [data_clean.iloc[1] - data_clean.iloc[0]]
        
        for t in range(1, n):
            new_level = alpha * data_clean.iloc[t] + (1 - alpha) * (level[-1] + trend[-1])
            new_trend = beta * (new_level - level[-1]) + (1 - beta) * trend[-1]
            level.append(new_level)
            trend.append(new_trend)
        
        forecast = []
        last_level = level[-1]
        last_trend = trend[-1]
        for i in range(1, forecast_days + 1):
            pred = last_level + i * last_trend
            # 確保預測值不為負數
            pred = max(0, pred)
            forecast.append(pred)
        
        return {
            'name': f'Holt-α={alpha},β={beta}',
            'forecast': forecast,
            'params': {'alpha': alpha, 'beta': beta},
            'last_value': forecast[0] if forecast else data.mean(),
            'model_type': '双指数平滑'
        }
    
    def seasonal_naive(self, data, period=7, forecast_days=None):
        """季節性朴素預測（保證非負）"""
        if forecast_days is None:
            forecast_days = self.forecast_days
        
        if len(data) < period:
            return None
        
        forecast = []
        for i in range(forecast_days):
            idx = -(period - (i % period)) if (i % period) != 0 else -period
            if abs(idx) <= len(data):
                pred = max(0, data.iloc[idx])  # 確保非負
            else:
                pred = max(0, data.mean())
            forecast.append(pred)
        
        return {
            'name': f'Seasonal-周期{period}天',
            'forecast': forecast,
            'params': {'period': period},
            'last_value': forecast[0],
            'model_type': '季节性预测'
        }
    
    # ========== 回歸分析模型 ==========
    
    def linear_regression_forecast(self, data, forecast_days=None, degree=1, alpha=None, model_type='linear'):
        """
        回归分析預測模型（线性/多项式/岭回归）
        
        Parameters:
        -----------
        data : Series
            時間序列數據
        forecast_days : int
            預測天數
        degree : int
            多項式回歸的次數（僅適用於多項式回歸）
        alpha : float
            正則化參數（適用於岭回归/Lasso）
        model_type : str
            回歸類型: 'linear', 'poly', 'ridge', 'lasso'
        """
        if forecast_days is None:
            forecast_days = self.forecast_days
        
        data_clean = data.clip(lower=0)
        
        if len(data_clean) < 3:
            avg = max(0, data_clean.mean())
            forecast = [avg] * forecast_days
            return {
                'name': f'{model_type}_回归(数据不足)',
                'forecast': forecast,
                'params': {'degree': degree, 'alpha': alpha},
                'last_value': avg,
                'model_type': f'{model_type}回归'
            }
        
        # 創建特徵：使用時間索引
        X = np.arange(len(data_clean)).reshape(-1, 1)
        y = data_clean.values
        
        try:
            if model_type == 'linear':
                model = LinearRegression()
                model_name = f'Linear_Reg'
            elif model_type == 'poly':
                model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                model_name = f'Poly_{degree}_Reg'
            elif model_type == 'ridge':
                model = Ridge(alpha=alpha if alpha else 1.0)
                model_name = f'Ridge_α={alpha if alpha else 1.0}'
            elif model_type == 'lasso':
                model = Lasso(alpha=alpha if alpha else 0.1, max_iter=1000)
                model_name = f'Lasso_α={alpha if alpha else 0.1}'
            else:
                model = LinearRegression()
                model_name = f'Linear_Reg'
            
            model.fit(X, y)
            
            # 預測未來
            future_X = np.arange(len(data_clean), len(data_clean) + forecast_days).reshape(-1, 1)
            forecast_raw = model.predict(future_X)
            forecast = [max(0, val) for val in forecast_raw]  # 確保非負
            
            # 計算R²來評估擬合度
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            return {
                'name': model_name,
                'forecast': forecast,
                'params': {'degree': degree, 'alpha': alpha, 'r2': round(r2, 3)},
                'last_value': forecast[0],
                'model_type': f'{model_type}回归',
                'r2_score': r2
            }
        except Exception as e:
            print(f"   ⚠️ 回歸模型訓練失敗: {e}")
            avg = max(0, data_clean.mean())
            forecast = [avg] * forecast_days
            return {
                'name': f'{model_type}_回归(回退)',
                'forecast': forecast,
                'params': {'degree': degree, 'alpha': alpha},
                'last_value': avg,
                'model_type': f'{model_type}回归'
            }
    
    def arima_like_auto_regression(self, data, lags=7, forecast_days=None):
        """
        自回歸模型 (AR-like)
        使用過去多天的值來預測未來
        
        Parameters:
        -----------
        data : Series
            時間序列數據
        lags : int
            使用的滯後天數
        forecast_days : int
            預測天數
        """
        if forecast_days is None:
            forecast_days = self.forecast_days
        
        data_clean = data.clip(lower=0).values
        
        if len(data_clean) < lags + 2:
            avg = max(0, np.mean(data_clean))
            forecast = [avg] * forecast_days
            return {
                'name': f'AR-{lags}(数据不足)',
                'forecast': forecast,
                'params': {'lags': lags},
                'last_value': avg,
                'model_type': '自回归'
            }
        
        # 創建特徵矩陣：使用過去lags天的值作為特徵
        X = []
        y = []
        for i in range(lags, len(data_clean)):
            X.append(data_clean[i-lags:i])
            y.append(data_clean[i])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < 2:
            avg = np.mean(data_clean[-lags:])
            forecast = [max(0, avg)] * forecast_days
            return {
                'name': f'AR-{lags}(样本不足)',
                'forecast': forecast,
                'params': {'lags': lags},
                'last_value': avg,
                'model_type': '自回归'
            }
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            
            # 遞迴預測
            forecast = []
            last_values = list(data_clean[-lags:])
            
            for _ in range(forecast_days):
                # 使用最後lags個值進行預測
                X_pred = np.array(last_values[-lags:]).reshape(1, -1)
                pred = model.predict(X_pred)[0]
                pred = max(0, pred)
                forecast.append(pred)
                last_values.append(pred)
            
            # 計算驗證集表現
            mae = np.mean(np.abs(y - model.predict(X))) if len(y) > 0 else float('inf')
            
            return {
                'name': f'AR-{lags}',
                'forecast': forecast,
                'params': {'lags': lags, 'val_mae': round(mae, 0)},
                'last_value': forecast[0],
                'model_type': '自回归'
            }
        except Exception as e:
            print(f"   ⚠️ 自回歸模型訓練失敗: {e}")
            avg = max(0, np.mean(data_clean[-lags:]))
            forecast = [avg] * forecast_days
            return {
                'name': f'AR-{lags}(回退)',
                'forecast': forecast,
                'params': {'lags': lags},
                'last_value': avg,
                'model_type': '自回归'
            }
    
    def moving_average_regression(self, data, window=7, forecast_days=None):
        """
        移動平均回歸 - 使用過去窗口期的平均趨勢
        
        Parameters:
        -----------
        data : Series
            時間序列數據
        window : int
            移動窗口大小
        forecast_days : int
            預測天數
        """
        if forecast_days is None:
            forecast_days = self.forecast_days
        
        data_clean = data.clip(lower=0)
        
        if len(data_clean) < window:
            avg = max(0, data_clean.mean())
            forecast = [avg] * forecast_days
            return {
                'name': f'MA_Reg-{window}(数据不足)',
                'forecast': forecast,
                'params': {'window': window},
                'last_value': avg,
                'model_type': '移动平均回归'
            }
        
        # 計算移動平均
        ma = data_clean.rolling(window=window, min_periods=1).mean()
        
        # 使用最近window天的移動平均值擬合線性趨勢
        recent_ma = ma.iloc[-window:].values
        X = np.arange(len(recent_ma)).reshape(-1, 1)
        y = recent_ma
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            
            # 預測未來
            future_X = np.arange(len(recent_ma), len(recent_ma) + forecast_days).reshape(-1, 1)
            forecast_raw = model.predict(future_X)
            forecast = [max(0, val) for val in forecast_raw]
            
            return {
                'name': f'MA_Trend-{window}',
                'forecast': forecast,
                'params': {'window': window, 'slope': round(model.coef_[0], 2)},
                'last_value': forecast[0],
                'model_type': '移动平均回归'
            }
        except Exception as e:
            print(f"   ⚠️ 移動平均回歸失敗: {e}")
            last_ma = max(0, ma.iloc[-1])
            forecast = [last_ma] * forecast_days
            return {
                'name': f'MA_Reg-{window}(回退)',
                'forecast': forecast,
                'params': {'window': window},
                'last_value': last_ma,
                'model_type': '移动平均回归'
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
            # 避免除以零
            mape = np.mean(np.abs((test_data.values - predictions) / np.maximum(test_data.values, 1))) * 100
            
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
        """訓練所有定義的模型（包含回歸模型）"""
        print("\n" + "=" * 80)
        print("🤖 初始化預測模型 (排除星期日)")
        print("=" * 80)
        
        models_config = [
            # 移動平均模型
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
            
            # 回歸分析模型
            {'func': self.linear_regression_forecast, 'params': {'model_type': 'linear'}, 'name': 'Linear_Regression'},
            {'func': self.linear_regression_forecast, 'params': {'model_type': 'poly', 'degree': 2}, 'name': 'Poly2_Regression'},
            {'func': self.linear_regression_forecast, 'params': {'model_type': 'poly', 'degree': 3}, 'name': 'Poly3_Regression'},
            {'func': self.linear_regression_forecast, 'params': {'model_type': 'ridge', 'alpha': 1.0}, 'name': 'Ridge_Regression'},
            {'func': self.linear_regression_forecast, 'params': {'model_type': 'ridge', 'alpha': 10.0}, 'name': 'Ridge_α10_Regression'},
            {'func': self.linear_regression_forecast, 'params': {'model_type': 'lasso', 'alpha': 0.1}, 'name': 'Lasso_Regression'},
            {'func': self.arima_like_auto_regression, 'params': {'lags': 3}, 'name': 'AR-3'},
            {'func': self.arima_like_auto_regression, 'params': {'lags': 7}, 'name': 'AR-7'},
            {'func': self.arima_like_auto_regression, 'params': {'lags': 14}, 'name': 'AR-14'},
            {'func': self.moving_average_regression, 'params': {'window': 7}, 'name': 'MA_Trend-7'},
            {'func': self.moving_average_regression, 'params': {'window': 14}, 'name': 'MA_Trend-14'},
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
        """對預測結果應用星期日過濾，並確保預測值不為負"""
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
            if model is None:
                continue
            filtered_forecast = []
            workday_idx = 0
            for forecast_date in self.all_forecast_dates:
                if forecast_date.weekday() == 6:  # 星期日
                    filtered_forecast.append(0)
                else:
                    if workday_idx < len(model['forecast']):
                        # 確保預測值不為負數
                        pred_value = max(0, model['forecast'][workday_idx])
                        filtered_forecast.append(pred_value)
                        workday_idx += 1
                    else:
                        filtered_forecast.append(0)
            
            model['original_forecast'] = model['forecast']
            model['forecast'] = filtered_forecast
        
        print(f"   ✅ 已對 {len(self.all_models)} 個模型應用星期日過濾及負值修正")
    
    def get_comparison_dataframe(self):
        """生成模型對比數據表"""
        comparison_data = []
        for model in self.all_models:
            if model is None:
                continue
            pred_total = sum(model['forecast'])
            pred_daily = pred_total / self.forecast_days if self.forecast_days > 0 else 0
            hist_daily = self.daily_total.mean() if len(self.daily_total) > 0 else 0
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
    
    # ========== 效能評估方法 ==========
    
    def load_actual_production_data(self, production_file_path=None):
        """
        載入實際生產數據用於評估
        
        Parameters:
        -----------
        production_file_path : str, optional
            實際生產數據Excel文件路徑，若為None則使用與訓練數據相同的路徑
        """
        if production_file_path is None:
            production_file_path = self.file_path.replace('.xlsx', ' - 複製.xlsx')
        
        print("\n" + "=" * 80)
        print("📊 載入實際生產數據進行效能評估")
        print("=" * 80)
        print(f"📂 讀取檔案: {production_file_path}")
        
        try:
            df_production = pd.read_excel(production_file_path, sheet_name='生产每日交货清单', skipfooter=1)
            df_production["开单日期"] = pd.to_datetime(df_production["开单日期"], format='%Y-%m-%d')
            df_production = df_production.loc[:, ["开单日期", "款号", "发出部门", "交货数"]]
            
            # 篩選車縫部門和指定款號
            df_filtered = df_production[
                df_production['发出部门'].str.contains('車縫', na=False) &
                df_production['款号'].isin(self.items)
            ].copy()
            
            self.actual_production_data = df_filtered.groupby('开单日期')['交货数'].sum().reset_index()
            self.actual_production_data.columns = ['日期', '實際交貨數']
            
            print(f"✅ 成功讀取 {len(self.actual_production_data)} 筆實際生產數據")
            print(f"   - 日期範圍: {self.actual_production_data['日期'].min()} ~ {self.actual_production_data['日期'].max()}")
            
            return self.actual_production_data
            
        except Exception as e:
            print(f"⚠️ 無法載入實際生產數據: {e}")
            print("   將僅使用訓練數據中的驗證集進行評估")
            self.actual_production_data = None
            return None
    
    def evaluate_with_actual_data(self):
        """
        使用實際生產數據評估模型預測準確度
        """
        if self.actual_production_data is None:
            print("\n⚠️ 無實際生產數據，跳過效能評估")
            return None
        
        print("\n" + "=" * 80)
        print("📈 模型效能評估（使用實際生產數據）")
        print("=" * 80)
        
        # 準備預測數據
        forecast_df = pd.DataFrame({
            '預測日期': self.all_forecast_dates,
            '星期': [d.weekday() for d in self.all_forecast_dates]
        })
        
        for model in self.all_models:
            if model is not None:
                forecast_df[model['name']] = model['forecast']
        
        # 合併預測與實際數據
        merged_df = forecast_df.merge(
            self.actual_production_data,
            left_on='預測日期',
            right_on='日期',
            how='left'
        )
        
        # 只保留有實際觀測數據的記錄
        valid_df = merged_df[merged_df['實際交貨數'].notna() & (merged_df['實際交貨數'] > 0)].copy()
        
        if len(valid_df) == 0:
            print("⚠️ 沒有可用的實際觀測數據進行評估")
            return None
        
        print(f"\n📋 有效觀測數據筆數: {len(valid_df)} 筆")
        print(f"   - 日期範圍: {valid_df['預測日期'].min()} ~ {valid_df['預測日期'].max()}")
        
        # 計算各模型誤差指標
        evaluation_results = []
        
        for model in self.all_models:
            if model is None:
                continue
            y_true = valid_df['實際交貨數'].values
            y_pred = valid_df[model['name']].values
            
            # 計算誤差指標
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = sqrt(np.mean((y_true - y_pred) ** 2))
            # 避免除以零
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
            
            # R平方
            r2 = r2_score(y_true, y_pred)
            
            # 偏誤
            bias = np.mean(y_pred) - np.mean(y_true)
            
            evaluation_results.append({
                '模型名稱': model['name'],
                '模型類型': model['model_type'],
                'MAE(件)': round(mae, 0),
                'RMSE(件)': round(rmse, 0),
                'MAPE(%)': round(mape, 1),
                'R²': round(r2, 3),
                '偏誤(Bias)': round(bias, 0),
                '平均預測': round(np.mean(y_pred), 0),
                '實際平均': round(np.mean(y_true), 0)
            })
        
        self.evaluation_results = pd.DataFrame(evaluation_results)
        self.evaluation_results = self.evaluation_results.sort_values('MAE(件)')
        
        print("\n🏆 模型效能排名（按MAE由低到高）:")
        print(self.evaluation_results[['模型名稱', 'MAE(件)', 'RMSE(件)', 'MAPE(%)', 'R²', '偏誤(Bias)']].head(15).to_string(index=False))
        
        return self.evaluation_results
    
    def categorize_evaluation_models(self):
        """對評估結果進行模型分類統計"""
        if self.evaluation_results is None:
            return None
        
        print("\n" + "=" * 80)
        print("📊 各類型模型效能對比")
        print("=" * 80)
        
        model_categories = {
            '移動平均': ['SMA-3', 'SMA-7', 'SMA-15', 'SMA-30'],
            '加權移動平均': ['WMA-7', 'WMA-15'],
            '指數平滑': ['ES-α=0.3', 'ES-α=0.5', 'ES-α=0.7'],
            '雙指數平滑': ['Holt-α=0.5,β=0.3', 'Holt-α=0.6,β=0.2'],
            '季節性預測': ['Seasonal-周期7天', 'Seasonal-周期15天', 'Seasonal-周期30天'],
            '線性回歸': ['Linear_Regression'],
            '多項式回歸': ['Poly2_Regression', 'Poly3_Regression'],
            '正則化回歸': ['Ridge_Regression', 'Ridge_α10_Regression', 'Lasso_Regression'],
            '自回歸': ['AR-3', 'AR-7', 'AR-14'],
            '移動平均回歸': ['MA_Trend-7', 'MA_Trend-14']
        }
        
        category_results = []
        for category, models in model_categories.items():
            category_models = self.evaluation_results[self.evaluation_results['模型名稱'].isin(models)]
            if len(category_models) > 0:
                category_results.append({
                    '模型類型': category,
                    '平均MAE': round(category_models['MAE(件)'].mean(), 0),
                    '平均MAPE(%)': round(category_models['MAPE(%)'].mean(), 1),
                    '最佳模型': category_models.iloc[category_models['MAE(件)'].argmin()]['模型名稱'],
                    '最佳MAE': category_models['MAE(件)'].min()
                })
        
        self.category_evaluation = pd.DataFrame(category_results)
        self.category_evaluation = self.category_evaluation.sort_values('平均MAE')
        print(self.category_evaluation.to_string(index=False))
        
        return self.category_evaluation
    
    def plot_evaluation_charts(self):
        """生成效能評估圖表"""
        if self.evaluation_results is None or len(self.evaluation_results) == 0:
            print("⚠️ 無評估結果，跳過圖表生成")
            return
        
        best_model_row = self.evaluation_results.iloc[0]
        best_model_name = best_model_row['模型名稱']
        best_mae = best_model_row['MAE(件)']
        best_mape = best_model_row['MAPE(%)']
        best_r2 = best_model_row['R²']
        
        # 準備數據
        forecast_df = pd.DataFrame({'預測日期': self.all_forecast_dates})
        for model in self.all_models:
            if model is not None:
                forecast_df[model['name']] = model['forecast']
        
        merged_df = forecast_df.merge(
            self.actual_production_data,
            left_on='預測日期',
            right_on='日期',
            how='left'
        )
        valid_df = merged_df[merged_df['實際交貨數'].notna() & (merged_df['實際交貨數'] > 0)].copy()
        
        if len(valid_df) == 0:
            print("⚠️ 無有效數據生成圖表")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'車縫組 - {self.items[:3] if len(self.items) > 3 else self.items}...預測模型效能評估 (最佳模型: {best_model_name})', fontsize=14, fontweight='bold')
        
        # 圖1: 最佳模型預測 vs 實際值
        ax = axes[0, 0]
        df_sorted = valid_df.sort_values('預測日期')
        ax.plot(df_sorted['預測日期'], df_sorted['實際交貨數'], 'b-o', linewidth=2, markersize=6, label='實際觀測值')
        ax.plot(df_sorted['預測日期'], df_sorted[best_model_name], 'r--s', linewidth=2, markersize=6, label=f'{best_model_name} 預測值')
        ax.set_title(f'最佳模型: {best_model_name}\nMAE={best_mae:.0f}, MAPE={best_mape:.1f}%, R²={best_r2:.3f}', 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('交貨數 (件)')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 圖2: 誤差柱狀圖
        ax = axes[0, 1]
        top_models = self.evaluation_results.head(10)
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(top_models))]
        bars = ax.barh(top_models['模型名稱'], top_models['MAE(件)'], color=colors, edgecolor='black')
        ax.set_title('模型誤差對比 (MAE - 越低越好)', fontsize=11, fontweight='bold')
        ax.set_xlabel('平均絕對誤差 (件)')
        ax.axvline(x=top_models['MAE(件)'].mean(), color='red', linestyle='--', alpha=0.7, 
                   label=f'平均MAE: {top_models["MAE(件)"].mean():.0f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, top_models['MAE(件)']):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, f'{val:.0f}', 
                   va='center', fontsize=9)
        
        # 圖3: 預測誤差分布
        ax = axes[1, 0]
        errors = valid_df['實際交貨數'].values - valid_df[best_model_name].values
        ax.hist(errors, bins=15, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零誤差線')
        ax.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
                   label=f'平均誤差: {np.mean(errors):.0f}')
        ax.set_title(f'{best_model_name} 預測誤差分布', fontsize=11, fontweight='bold')
        ax.set_xlabel('預測誤差 (件)')
        ax.set_ylabel('頻次')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 圖4: 散點圖
        ax = axes[1, 1]
        ax.scatter(valid_df['實際交貨數'], valid_df[best_model_name], alpha=0.7, s=80, 
                   c='steelblue', edgecolors='navy')
        max_val = max(valid_df['實際交貨數'].max(), valid_df[best_model_name].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美預測線')
        ax.set_title(f'{best_model_name} - 預測值 vs 實際值', fontsize=11, fontweight='bold')
        ax.set_xlabel('實際交貨數 (件)')
        ax.set_ylabel('預測交貨數 (件)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '模型效能評估對比圖.png'), dpi=150, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ 效能評估圖表已儲存")
    
    def plot_all_models_comparison(self):
        """生成所有模型預測對比圖"""
        if self.actual_production_data is None:
            return
        
        forecast_df = pd.DataFrame({'預測日期': self.all_forecast_dates})
        for model in self.all_models:
            if model is not None:
                forecast_df[model['name']] = model['forecast']
        
        merged_df = forecast_df.merge(
            self.actual_production_data,
            left_on='預測日期',
            right_on='日期',
            how='left'
        )
        valid_df = merged_df[merged_df['實際交貨數'].notna() & (merged_df['實際交貨數'] > 0)].copy()
        
        if len(valid_df) == 0:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('車縫組 - 所有模型預測 vs 實際對比', fontsize=14, fontweight='bold')
        
        for idx, model in enumerate(self.all_models[:6]):
            if model is None:
                continue
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            df_sorted = valid_df.sort_values('預測日期')
            ax.plot(df_sorted['預測日期'], df_sorted['實際交貨數'], 'b-o', linewidth=2, markersize=5, label='實際觀測值')
            ax.plot(df_sorted['預測日期'], df_sorted[model['name']], 'r--s', linewidth=2, markersize=5, label=f'{model["name"]}')
            
            model_mae = self.evaluation_results[self.evaluation_results['模型名稱'] == model['name']]['MAE(件)'].values[0]
            ax.set_title(f'{model["name"]} (MAE: {model_mae:.0f}件)', fontsize=10, fontweight='bold')
            ax.set_xlabel('日期')
            ax.set_ylabel('交貨數 (件)')
            ax.legend(fontsize=8)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '所有模型預測對比圖.png'), dpi=150, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ 所有模型對比圖已儲存")
    
    def print_evaluation_recommendations(self):
        """打印模型推薦建議"""
        if self.evaluation_results is None or len(self.evaluation_results) == 0:
            return
        
        best_model = self.evaluation_results.iloc[0]['模型名稱']
        best_mae = self.evaluation_results.iloc[0]['MAE(件)']
        best_mape = self.evaluation_results.iloc[0]['MAPE(%)']
        second_best = self.evaluation_results.iloc[1]['模型名稱'] if len(self.evaluation_results) > 1 else None
        
        print("\n" + "=" * 80)
        print("💡 模型推薦建議")
        print("=" * 80)
        
        print(f"""
📌 最佳模型: {best_model}
   - 優點: 預測誤差最小，準確度最高
   - 適用場景: 一般生產規劃與預測
   - MAE: {best_mae:.0f}件 | MAPE: {best_mape:.1f}%

📌 次佳模型: {second_best}
   - 可作為備用方案或模型集成使用

📌 各類型模型推薦:
""")
        
        if self.category_evaluation is not None:
            for _, row in self.category_evaluation.iterrows():
                print(f"   • {row['模型類型']}: {row['最佳模型']} (MAE: {row['最佳MAE']:.0f}件)")
        
        # 找出SMA模型
        sma_models = self.evaluation_results[self.evaluation_results['模型名稱'].str.contains('SMA')]
        best_sma = sma_models.iloc[0]['模型名稱'] if len(sma_models) > 0 else 'SMA模型'
        
        # 找出季節性模型
        seasonal_models = self.category_evaluation[self.category_evaluation['模型類型'] == '季節性預測'] if self.category_evaluation is not None and len(self.category_evaluation) > 0 else pd.DataFrame()
        best_seasonal = seasonal_models['最佳模型'].values[0] if len(seasonal_models) > 0 else '季節性模型'
        
        # 找出回歸模型
        regression_models = self.evaluation_results[self.evaluation_results['模型名稱'].str.contains('Reg|AR-')]
        best_regression = regression_models.iloc[0]['模型名稱'] if len(regression_models) > 0 else '回歸模型'
        
        print(f"""
📊 模型選擇建議:
   1. 優先使用 {best_model} 進行預測
   2. 若需要預測穩定性，可考慮 {best_sma}
   3. 若數據有季節性趨勢，可參考 {best_seasonal}
   4. 若需要捕捉長期趨勢，可考慮回歸模型: {best_regression}
   5. 建議使用多模型集成（取平均）來降低單一模型風險
""")
    
    def save_evaluation_excel(self):
        """保存效能評估Excel報告"""
        if self.evaluation_results is None:
            return
        
        excel_file = os.path.join(self.output_folder, '模型效能評估報告.xlsx')
        
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 模型排名表
                self.evaluation_results.to_excel(writer, sheet_name='模型排名', index=False)
                
                # 各模型類型匯總
                if self.category_evaluation is not None:
                    self.category_evaluation.to_excel(writer, sheet_name='模型類型匯總', index=False)
                
                # 預測vs實際明細
                forecast_df = pd.DataFrame({'預測日期': self.all_forecast_dates})
                for model in self.all_models:
                    if model is not None:
                        forecast_df[model['name']] = model['forecast']
                
                merged_df = forecast_df.merge(
                    self.actual_production_data,
                    left_on='預測日期',
                    right_on='日期',
                    how='left'
                )
                detail_df = merged_df[merged_df['實際交貨數'].notna() & (merged_df['實際交貨數'] > 0)].copy()
                detail_df['星期'] = detail_df['預測日期'].dt.dayofweek
                detail_df.to_excel(writer, sheet_name='預測vs實際明細', index=False)
                
                # 模型推薦
                best_model = self.evaluation_results.iloc[0]['模型名稱']
                second_best = self.evaluation_results.iloc[1]['模型名稱'] if len(self.evaluation_results) > 1 else 'N/A'
                sma_models = self.evaluation_results[self.evaluation_results['模型名稱'].str.contains('SMA')]
                best_sma = sma_models.iloc[0]['模型名稱'] if len(sma_models) > 0 else 'SMA-7'
                
                recommendation = pd.DataFrame({
                    '推薦類型': ['最佳準確度', '備用方案', '最穩定(移動平均)', '保守預測', '樂觀預測'],
                    '推薦模型': [
                        best_model,
                        second_best,
                        best_sma,
                        min([m for m in self.all_models if m is not None], key=lambda x: sum(x['forecast']))['name'],
                        max([m for m in self.all_models if m is not None], key=lambda x: sum(x['forecast']))['name']
                    ],
                    'MAE(件)': [
                        self.evaluation_results.iloc[0]['MAE(件)'],
                        self.evaluation_results.iloc[1]['MAE(件)'] if len(self.evaluation_results) > 1 else 'N/A',
                        sma_models.iloc[0]['MAE(件)'] if len(sma_models) > 0 else 'N/A',
                        'N/A', 'N/A'
                    ]
                })
                recommendation.to_excel(writer, sheet_name='模型推薦', index=False)
            
            print(f"\n✅ Excel評估報告已儲存: {excel_file}")
        except Exception as e:
            print(f"\n⚠️ Excel儲存失敗: {e}")
    
    # ========== 圖表生成方法（原有功能） ==========
    
    def plot_comparison_charts(self):
        """生成多模型對比圖表"""
        print("\n" + "=" * 80)
        print("📈 生成多模型對比圖表")
        print("=" * 80)
        
        # 過濾掉 None 模型
        valid_models = [m for m in self.all_models if m is not None]
        if not valid_models:
            print("⚠️ 沒有有效的模型進行對比")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'車縫組{self.items[:3] if len(self.items) > 3 else self.items}* - 多模型預測對比分析 (排除星期日)', fontsize=16, fontweight='bold')
        
        # 安全獲取 MAE 的輔助函數
        def get_model_mae(model):
            evaluation = model.get('evaluation')
            if evaluation and isinstance(evaluation, dict):
                return evaluation.get('mae', float('inf'))
            return float('inf')
        
        best_model = min(valid_models, key=get_model_mae)
        
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
        
        if non_sunday_dates:
            ax.plot(non_sunday_dates, non_sunday_values, 'r--o', linewidth=2, markersize=4, 
                    label=f'最佳模型: {best_model["name"]}')
        if sunday_dates:
            ax.scatter(sunday_dates, [0]*len(sunday_dates), color='gray', marker='x', s=50, 
                      label='星期日(不生產)')
        
        ax.axhline(y=self.daily_total.mean(), color='blue', linestyle=':', alpha=0.5, 
                   label=f'歷史日均: {self.daily_total.mean():.0f}')
        ax.axhline(y=sum(best_model['forecast'])/self.forecast_days, color='red', linestyle=':', alpha=0.5, 
                   label=f'預測日均(含周日): {sum(best_model["forecast"])/self.forecast_days:.0f}')
        
        # 安全獲取評估值
        best_eval = best_model.get('evaluation')
        best_mae = best_eval.get('mae', 'N/A') if best_eval else 'N/A'
        best_mape = best_eval.get('mape', 'N/A') if best_eval else 'N/A'
        
        title_text = f'最佳模型預測 - {best_model["name"]}'
        if best_mae != 'N/A' and best_mape != 'N/A':
            title_text += f'\nMAE: {best_mae:.0f} | MAPE: {best_mape:.1f}%'
        ax.set_title(title_text, fontsize=11, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('交貨數 (件)')
        ax.legend(loc='upper left', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 2. 模型誤差對比
        ax = axes[0, 1]
        models_with_mae = [(m['name'], m.get('evaluation', {}).get('mae', 0) if m.get('evaluation') else 0) 
                          for m in valid_models]
        models_with_mae = [(name, mae) for name, mae in models_with_mae if mae > 0]  # 只保留有評估值的模型
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
        all_predictions = [m['forecast'] for m in valid_models]
        pred_array = np.array(all_predictions)
        pred_mean = pred_array.mean(axis=0)
        pred_min = pred_array.min(axis=0)
        pred_max = pred_array.max(axis=0)
        
        ax.fill_between(range(self.forecast_days), pred_min, pred_max, alpha=0.3, color='gray', 
                       label='預測範圍(最小-最大)')
        ax.plot(range(self.forecast_days), pred_mean, 'b-', linewidth=2, label='平均預測')
        ax.plot(range(self.forecast_days), [self.daily_total.mean()] * self.forecast_days, 'r--', 
               label='歷史日均線')
        ax.set_title(f'多模型預測範圍 (共{len(valid_models)}個模型, 排除星期日)', fontsize=11, fontweight='bold')
        ax.set_xlabel('預測天數')
        ax.set_ylabel('預測交貨數 (件)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 最佳3個模型預測對比
        ax = axes[1, 0]
        top_3 = sorted(valid_models, key=get_model_mae)[:3]
        colors = ['red', 'green', 'orange']
        for i, model in enumerate(top_3):
            non_sunday_vals = [model['forecast'][j] for j in range(self.forecast_days) 
                              if self.all_forecast_dates[j].weekday() != 6]
            if non_sunday_dates:
                # 安全獲取模型MAE
                model_eval = model.get('evaluation')
                model_mae = model_eval.get('mae', 0) if model_eval else 0
                ax.plot(non_sunday_dates, non_sunday_vals, color=colors[i], linewidth=2, 
                        label=f"{model['name']} (MAE:{model_mae:.0f})")
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
        for m in valid_models:
            evaluation = m.get('evaluation')
            if evaluation and isinstance(evaluation, dict):
                mtype = m['model_type']
                if mtype not in model_types:
                    model_types[mtype] = []
                model_types[mtype].append(evaluation.get('mae', 0))
        
        if model_types:
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
        
        # 6. 星期日影響分析 - 修復索引問題
        ax = axes[1, 2]
        daily_total_all_with_weekday = self.df_filtered.copy()
        daily_total_all_with_weekday['星期'] = daily_total_all_with_weekday['开单日期'].dt.dayofweek
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        
        # 計算各星期平均交貨量
        weekday_avg = daily_total_all_with_weekday.groupby('星期')['交货数'].mean()
        
        # 確保所有星期都有數據（補齊缺少的星期）
        for i in range(7):
            if i not in weekday_avg.index:
                weekday_avg[i] = 0
        
        # 重新排序並設置索引名稱
        weekday_avg = weekday_avg.reindex(range(7))
        weekday_avg.index = weekday_names
        
        # 設定顏色（星期日用不同顏色）
        colors_bar = ['steelblue'] * 6 + ['lightcoral']
        bars = ax.bar(weekday_names, weekday_avg.values, color=colors_bar, edgecolor='black')
        ax.set_title('各星期平均交貨量分析', fontsize=11, fontweight='bold')
        ax.set_xlabel('星期')
        ax.set_ylabel('平均交貨數 (件)')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, weekday_avg.values):
            if val > 0:
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
        
        valid_models = [m for m in self.all_models if m is not None]
        
        for i, model in enumerate(valid_models[:max_models]):
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
        
        print(f"   ✅ 已生成 {min(max_models, len(valid_models))} 個詳細預測圖")
    
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
                    if model is not None:
                        all_forecasts[model['name']] = [round(v) for v in model['forecast']]
                all_forecasts.to_excel(writer, sheet_name='所有模型預測匯總', index=False)
                
                sunday_count = sum(1 for d in self.all_forecast_dates if d.weekday() == 6)
                
                # 安全獲取評估值
                best_eval = best_model.get('evaluation')
                best_mae_str = f"{best_eval['mae']:.0f}" if best_eval else 'N/A'
                best_mape_str = f"{best_eval['mape']:.1f}%" if best_eval else 'N/A'
                
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
                        len([m for m in self.all_models if m is not None]),
                        best_model['name'],
                        best_mae_str,
                        best_mape_str,
                        f"{pred_mean.mean():.0f}",
                        f"{pred_min.mean():.0f} ~ {pred_max.mean():.0f}",
                        '星期日預測值設為0（不生產）'
                    ]
                })
                summary.to_excel(writer, sheet_name='統計摘要', index=False)
                
                valid_models = [m for m in self.all_models if m is not None]
                recommendation = pd.DataFrame({
                    '使用場景': ['短期波動大', '一般生產規劃', '平穩生產', '保守預測', '樂觀預測'],
                    '推薦模型': [
                        'SMA-3 或 ES-α=0.7',
                        'SMA-7 或 WMA-7',
                        'SMA-15 或 ES-α=0.3',
                        f'{min(valid_models, key=lambda x: sum(x["forecast"]))["name"]}' if valid_models else 'N/A',
                        f'{max(valid_models, key=lambda x: sum(x["forecast"]))["name"]}' if valid_models else 'N/A'
                    ],
                    '預測工作日日均(件)': [
                        'N/A', 'N/A', 'N/A',
                        f"{sum(min(valid_models, key=lambda x: sum(x['forecast']))['forecast'])/self.forecast_days:.0f}" if valid_models else 'N/A',
                        f"{sum(max(valid_models, key=lambda x: sum(x['forecast']))['forecast'])/self.forecast_days:.0f}" if valid_models else 'N/A'
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
        best_eval = best_model.get('evaluation')
        if best_eval:
            print(f"   - MAE (平均絕對誤差): {best_eval['mae']:.0f} 件")
            print(f"   - RMSE (均方根誤差): {best_eval['rmse']:.0f} 件")
            print(f"   - MAPE (平均絕對百分比誤差): {best_eval['mape']:.1f}%")
        print(f"   - 預測30天總產能: {sum(best_model['forecast']):,.0f} 件")
        print(f"   - 預測日均產能(含周日): {sum(best_model['forecast'])/self.forecast_days:.0f} 件/天")
        workday_forecast = [v for v in best_model['forecast'] if v > 0]
        if workday_forecast:
            print(f"   - 預測工作日日均產能: {np.mean(workday_forecast):.0f} 件/天")
        
        print(f"\n📊 多模型統計:")
        print(f"   - 模型總數: {len([m for m in self.all_models if m is not None])}")
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
        print("   ├── 模型效能評估報告.xlsx              (效能評估報告)")
        print("   ├── 模型效能評估對比圖.png              (效能評估圖表)")
        print("   ├── 所有模型預測對比圖.png              (模型對比圖)")
        print("   └── ...")
        
        print("\n" + "=" * 80)
        print("⚠️ 重要說明:")
        print(f"   1. 本報告使用{self.test_days}天歷史數據進行模型驗證")
        print("   2. 星期日不生產，預測值設為0")
        print("   3. 歷史訓練數據已排除星期日")
        print("   4. MAE越低代表模型越準確")
        print("   5. 預測工作日日均更能反映實際產能")
        print("   6. 所有預測值均已確保不為負數（Holt模型負值已修正）")
        print("   7. 新增回歸分析模型（線性、多項式、嶺回歸、自回歸等）")
        print("=" * 80)
        
        print("\n✨ 多模型預測對比報告生成完成！")
        print(f"📁 報告已儲存至: {os.path.abspath(self.output_folder)}")
    
    # ========== 主運行方法 ==========
    
    def run(self, production_file_path=None):
        """
        執行完整的預測流程
        
        Parameters:
        -----------
        production_file_path : str, optional
            實際生產數據文件路徑，用於效能評估
        """
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
        valid_models = [m for m in self.all_models if m is not None]
        if valid_models:
            all_predictions = [m['forecast'] for m in valid_models]
            pred_array = np.array(all_predictions)
            pred_min = pred_array.min(axis=0)
            pred_max = pred_array.max(axis=0)
            pred_mean = pred_array.mean(axis=0)
        else:
            pred_min = pred_max = pred_mean = np.zeros(self.forecast_days)
        
        sunday_count = sum(1 for d in self.all_forecast_dates if d.weekday() == 6)
        
        # 10. 保存Excel報告
        if best_model is not None:
            self.save_excel_report(best_model, comparison_df, pred_mean, pred_min, pred_max)
        
        # 11. 打印最終報告
        if best_model is not None:
            self.print_final_report(best_model, pred_mean, pred_min, pred_max, sunday_count)
        else:
            print("\n⚠️ 無法生成最終報告：沒有有效的模型")
        
        # 12. 效能評估（如果有實際數據）
        if production_file_path is not None or os.path.exists(self.file_path.replace('.xlsx', ' - 複製.xlsx')):
            try:
                self.load_actual_production_data(production_file_path)
                self.evaluate_with_actual_data()
                self.categorize_evaluation_models()
                self.plot_evaluation_charts()
                self.plot_all_models_comparison()
                self.print_evaluation_recommendations()
                self.save_evaluation_excel()
            except Exception as e:
                print(f"⚠️ 效能評估失敗: {e}")
                print("   將跳過效能評估部分")
        else:
            print("\n⚠️ 未找到實際生產數據文件，跳過效能評估")
            print("   如需進行效能評估，請準備文件名為 '生产每日交货清单 - 複製.xlsx' 的實際數據文件")
        
        return best_model


def get_防火料款號():
    wb_path = r"C:\Users\Dell\Desktop\git_demo\防火料款號.xlsx"
    sht = '表格資料'
    df = pd.read_excel(wb_path, sheet_name=sht)
    return df['item'].to_list()


# ============================================
# 使用範例
# ============================================
if __name__ == "__main__":
    # 配置參數
    # items = get_防火料款號()
    # items = ['P11774-8', 'P11774-7', 'P11775-7', 'P11775-8']  # 可以修改為需要的款號
    items = ['P12857-1','P12857-4', 'P12857-TD6', 'P12857-3A', 'P12857-TM10','P13040-3','P13040-3']
    # items = ['P13040-3', 'P13040-1']
    # items = ['P12673-5']
    file_path = r'C:\Users\Dell\Downloads\生产每日交货清单.xlsx'
    production_file_path = r'C:\Users\Dell\Downloads\生产每日交货清单(實際生產數據).xlsx'
    
    # 創建分析器實例並運行
    analyzer = SewingForecastAnalyzer(
        file_path=file_path,
        items=items,
        test_days=60,
        forecast_days=30,
        output_folder='车缝组_多模型预测对比报告(排除周日)'
    )
    
    # 執行完整分析（包含效能評估）
    best_model = analyzer.run(production_file_path)
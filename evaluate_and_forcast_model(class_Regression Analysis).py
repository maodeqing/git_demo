import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
from math import sqrt
from typing import List, Dict, Optional, Any, Union, Tuple
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
    
    def __init__(
        self, 
        file_path: str, 
        items: List[str], 
        test_days: int = 45, 
        forecast_days: int = 30, 
        output_folder: Optional[str] = None
    ) -> None:
        """
        初始化預測分析器
        
        Parameters:
        -----------
        file_path : str
            Excel文件路径
        items : list
            款号列表
        test_days : int
            验证集天数
        forecast_days : int
            预测天数
        output_folder : str
            输出文件夹名称，默认为自动生成
        """
        self.file_path: str = file_path
        self.items: List[str] = items
        self.test_days: int = test_days
        self.forecast_days: int = forecast_days
        
        # 设置输出文件夹
        if output_folder is None:
            self.output_folder: str = f'车缝组_多模型预测对比报告(排除周日)'
        else:
            self.output_folder: str = output_folder
        
        # 数据属性
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_filtered: Optional[pd.DataFrame] = None
        self.daily_total_all: Optional[pd.Series] = None
        self.daily_total: Optional[pd.Series] = None
        self.all_forecast_dates: Optional[List[datetime]] = None
        self.all_models: List[Optional[Dict[str, Any]]] = []
        
        # 评估相关属性
        self.actual_production_data: Optional[pd.DataFrame] = None
        self.evaluation_results: Optional[pd.DataFrame] = None
        self.category_evaluation: Optional[pd.DataFrame] = None
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建输出文件夹
        self._create_output_folder()
    
    def _create_output_folder(self) -> None:
        """创建输出文件夹"""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"📁 创建文件夹: {self.output_folder}")
    
    def load_data(self) -> 'SewingForecastAnalyzer':
        """读取并预处理Excel资料"""
        print("=" * 80)
        print(f"📊 车缝组 - 款号* 多模型预测对比报告 (排除星期日)")
        print("=" * 80)
        print(f"\n📂 读取文件: {self.file_path}")
        
        try:
            self.df_raw = pd.read_excel(self.file_path, sheet_name='生产每日交货清单')
            print(f"✅ 成功读取 {len(self.df_raw)} 笔资料")
        except FileNotFoundError:
            print(f"❌ 文件不存在: {self.file_path}")
            raise
        except Exception as e:
            print(f"❌ 读取失败: {e}")
            raise
        
        # 检查必要字段
        required_columns = ['开单日期', '发出部门', '交货数']
        for col in required_columns:
            if col not in self.df_raw.columns:
                raise ValueError(f"缺少必要字段: {col}")
        
        # 资料清理
        self.df_raw['开单日期'] = pd.to_datetime(self.df_raw['开单日期'])
        self.df_raw = self.df_raw.dropna(subset=['开单日期', '发出部门', '交货数'])
        self.df_raw = self.df_raw[self.df_raw['交货数'] > 0]
        self.df_raw = self.df_raw.sort_values('开单日期')
        
        return self
    
    def filter_data(self) -> 'SewingForecastAnalyzer':
        """筛选车缝部门和指定款号"""
        print("\n" + "=" * 80)
        print("🔍 筛选条件:")
        print("   - 发出部门 包含「车缝」")
        print(f"   - 款号 以下款号：{self.items}")
        print("   - 交货数 > 0")
        print("   - 排除星期日（星期天不预测）")
        print("=" * 80)
        
        assert self.df_raw is not None, "请先调用 load_data()"
        
        sewing_mask = self.df_raw['发出部门'].str.contains('車縫', na=False)
        df_sewing = self.df_raw[sewing_mask].copy()
        
        if '款号' in df_sewing.columns:
            mask = df_sewing['款号'].isin(self.items)
            self.df_filtered = df_sewing[mask].copy()
            print(f"\n✅ 筛选完成: {len(self.df_filtered)} 笔资料")
            print(f"   - 款号: {self.df_filtered['款号'].unique().tolist()}")
        else:
            self.df_filtered = df_sewing.copy()
            print(f"\n⚠️ 无款号字段，仅筛选车缝部门")
        
        if len(self.df_filtered) == 0:
            raise ValueError("没有符合条件的资料！")
        
        # 每日汇总（包含所有日期）
        self.daily_total_all = self.df_filtered.groupby('开单日期')['交货数'].sum()
        
        # 过滤掉星期日的数据（用于模型训练）
        self.df_filtered['星期幾'] = self.df_filtered['开单日期'].dt.dayofweek
        df_filtered_no_sunday = self.df_filtered[self.df_filtered['星期幾'] != 6].copy()
        self.daily_total = df_filtered_no_sunday.groupby('开单日期')['交货数'].sum()
        
        print(f"\n📊 历史资料统计:")
        print(f"   - 原始有交货天数: {len(self.daily_total_all)} 天")
        print(f"   - 排除星期日后天数: {len(self.daily_total)} 天")
        print(f"   - 排除星期日数: {len(self.daily_total_all) - len(self.daily_total)} 天")
        print(f"   - 总交货量: {self.daily_total.sum():,.0f} 件")
        print(f"   - 日均产能(仅工作日): {self.daily_total.mean():.0f} 件/天")
        
        return self
    
    def get_all_forecast_dates(self) -> List[datetime]:
        """获取包含星期日的完整预测日期列表"""
        assert self.daily_total is not None, "请先调用 filter_data()"
        
        last_date = self.daily_total.index.max()
        start_forecast_date = last_date + timedelta(days=1)
        self.all_forecast_dates = [start_forecast_date + timedelta(days=i) for i in range(self.forecast_days)]
        return self.all_forecast_dates
    
    # ========== 预测模型方法（已添加负值保护） ==========
    
    def simple_moving_average(
        self, 
        data: pd.Series, 
        window: int, 
        forecast_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """简单移动平均（保证非负）"""
        if forecast_days is None:
            forecast_days = self.forecast_days
        ma = data.rolling(window=window, min_periods=1).mean()
        last_ma = max(0, ma.iloc[-1])  # 确保非负
        forecast = [last_ma] * forecast_days
        
        return {
            'name': f'SMA-{window}',
            'forecast': forecast,
            'params': {'window': window},
            'last_value': last_ma,
            'model_type': '移动平均'
        }
    
    def weighted_moving_average(
        self, 
        data: pd.Series, 
        window: int, 
        forecast_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """加权移动平均（保证非负）"""
        if forecast_days is None:
            forecast_days = self.forecast_days
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        
        last_n = data.iloc[-window:].values if len(data) >= window else data.values
        if len(last_n) >= window:
            wma = np.sum(last_n * weights[-len(last_n):])
        else:
            wma = data.mean()
        
        wma = max(0, wma)  # 确保非负
        forecast = [wma] * forecast_days
        
        return {
            'name': f'WMA-{window}',
            'forecast': forecast,
            'params': {'window': window, 'weights': 'linear'},
            'last_value': wma,
            'model_type': '加权移动平均'
        }
    
    def exponential_smoothing(
        self, 
        data: pd.Series, 
        alpha: float, 
        forecast_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """指数平滑（保证非负）"""
        if forecast_days is None:
            forecast_days = self.forecast_days
        data_clean = data.clip(lower=0)
        result = [data_clean.iloc[0]]
        for t in range(1, len(data_clean)):
            result.append(alpha * data_clean.iloc[t] + (1 - alpha) * result[-1])
        
        last_smooth = max(0, result[-1])  # 确保非负
        forecast = [last_smooth] * forecast_days
        
        return {
            'name': f'ES-α={alpha}',
            'forecast': forecast,
            'params': {'alpha': alpha},
            'last_value': last_smooth,
            'model_type': '指数平滑'
        }
    
    def double_exponential_smoothing(
        self, 
        data: pd.Series, 
        alpha: float, 
        beta: float, 
        forecast_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Holt's 线性指数平滑（保证预测值不为负）"""
        if forecast_days is None:
            forecast_days = self.forecast_days
        n = len(data)
        
        # 确保数据非负
        data_clean = data.clip(lower=0)
        
        if n < 2:
            # 数据不足时返回简单平均值
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
            # 确保预测值不为负数
            pred = max(0, pred)
            forecast.append(pred)
        
        return {
            'name': f'Holt-α={alpha},β={beta}',
            'forecast': forecast,
            'params': {'alpha': alpha, 'beta': beta},
            'last_value': forecast[0] if forecast else data.mean(),
            'model_type': '双指数平滑'
        }
    
    def seasonal_naive(
        self, 
        data: pd.Series, 
        period: int = 7, 
        forecast_days: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """季节性朴素预测（保证非负）"""
        if forecast_days is None:
            forecast_days = self.forecast_days
        
        if len(data) < period:
            return None
        
        forecast = []
        for i in range(forecast_days):
            idx = -(period - (i % period)) if (i % period) != 0 else -period
            if abs(idx) <= len(data):
                pred = max(0, data.iloc[idx])  # 确保非负
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
    
    # ========== 回归分析模型 ==========
    
    def linear_regression_forecast(
        self, 
        data: pd.Series, 
        forecast_days: Optional[int] = None, 
        degree: int = 1, 
        alpha: Optional[float] = None, 
        model_type: str = 'linear'
    ) -> Dict[str, Any]:
        """
        回归分析预测模型（线性/多项式/岭回归）
        
        Parameters:
        -----------
        data : Series
            时间序列数据
        forecast_days : int
            预测天数
        degree : int
            多项式回归的次数（仅适用于多项式回归）
        alpha : float
            正则化参数（适用于岭回归/Lasso）
        model_type : str
            回归类型: 'linear', 'poly', 'ridge', 'lasso'
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
        
        # 创建特征：使用时间索引
        X = np.arange(len(data_clean)).reshape(-1, 1)
        y = data_clean.values
        
        try:
            if model_type == 'linear':
                model: Union[LinearRegression, Ridge, Lasso] = LinearRegression()
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
            
            # 预测未来
            future_X = np.arange(len(data_clean), len(data_clean) + forecast_days).reshape(-1, 1)
            forecast_raw = model.predict(future_X)
            forecast = [max(0, val) for val in forecast_raw]  # 确保非负
            
            # 计算R²来评估拟合度
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
            print(f"   ⚠️ 回归模型训练失败: {e}")
            avg = max(0, data_clean.mean())
            forecast = [avg] * forecast_days
            return {
                'name': f'{model_type}_回归(回退)',
                'forecast': forecast,
                'params': {'degree': degree, 'alpha': alpha},
                'last_value': avg,
                'model_type': f'{model_type}回归'
            }
    
    def arima_like_auto_regression(
        self, 
        data: pd.Series, 
        lags: int = 7, 
        forecast_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        自回归模型 (AR-like)
        使用过去多天的值来预测未来
        
        Parameters:
        -----------
        data : Series
            时间序列数据
        lags : int
            使用的滞后天数
        forecast_days : int
            预测天数
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
        
        # 创建特征矩阵：使用过去lags天的值作为特征
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
            
            # 递归预测
            forecast = []
            last_values = list(data_clean[-lags:])
            
            for _ in range(forecast_days):
                # 使用最后lags个值进行预测
                X_pred = np.array(last_values[-lags:]).reshape(1, -1)
                pred = model.predict(X_pred)[0]
                pred = max(0, pred)
                forecast.append(pred)
                last_values.append(pred)
            
            # 计算验证集表现
            mae = np.mean(np.abs(y - model.predict(X))) if len(y) > 0 else float('inf')
            
            return {
                'name': f'AR-{lags}',
                'forecast': forecast,
                'params': {'lags': lags, 'val_mae': round(mae, 0)},
                'last_value': forecast[0],
                'model_type': '自回归'
            }
        except Exception as e:
            print(f"   ⚠️ 自回归模型训练失败: {e}")
            avg = max(0, np.mean(data_clean[-lags:]))
            forecast = [avg] * forecast_days
            return {
                'name': f'AR-{lags}(回退)',
                'forecast': forecast,
                'params': {'lags': lags},
                'last_value': avg,
                'model_type': '自回归'
            }
    
    def moving_average_regression(
        self, 
        data: pd.Series, 
        window: int = 7, 
        forecast_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        移动平均回归 - 使用过去窗口期的平均趋势
        
        Parameters:
        -----------
        data : Series
            时间序列数据
        window : int
            移动窗口大小
        forecast_days : int
            预测天数
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
        
        # 计算移动平均
        ma = data_clean.rolling(window=window, min_periods=1).mean()
        
        # 使用最近window天的移动平均值拟合线性趋势
        recent_ma = ma.iloc[-window:].values
        X = np.arange(len(recent_ma)).reshape(-1, 1)
        y = recent_ma
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            
            # 预测未来
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
            print(f"   ⚠️ 移动平均回归失败: {e}")
            last_ma = max(0, ma.iloc[-1])
            forecast = [last_ma] * forecast_days
            return {
                'name': f'MA_Reg-{window}(回退)',
                'forecast': forecast,
                'params': {'window': window},
                'last_value': last_ma,
                'model_type': '移动平均回归'
            }
    
    def evaluate_model(
        self, 
        model_func, 
        **kwargs
    ) -> Optional[Dict[str, Union[float, np.ndarray]]]:
        """评估模型在验证集上的表现"""
        assert self.daily_total is not None, "请先调用 filter_data()"
        
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
            print(f"   ⚠️ 模型评估失败: {e}")
            return None
    
    def train_models(self) -> List[Optional[Dict[str, Any]]]:
        """训练所有定义的模型（包含回归模型）"""
        print("\n" + "=" * 80)
        print("🤖 初始化预测模型 (排除星期日)")
        print("=" * 80)
        
        models_config = [
            # 移动平均模型
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
            
            # 回归分析模型
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
        
        assert self.daily_total is not None, "请先调用 filter_data()"
        
        self.all_models = []
        print("\n📊 训练模型中...")
        
        for config in models_config:
            try:
                result = config['func'](self.daily_total, forecast_days=self.forecast_days, **config['params'])
                
                if result is not None:
                    evaluation = self.evaluate_model(config['func'], **config['params'])
                    result['evaluation'] = evaluation
                    result['config'] = config
                    self.all_models.append(result)
                    print(f"   ✅ {result['name']} - 训练完成")
            except Exception as e:
                print(f"   ⚠️ {config['name']} 训练失败: {e}")
        
        print(f"\n✅ 成功训练 {len(self.all_models)} 个模型")
        return self.all_models
    
    def apply_sunday_filter(self) -> None:
        """对预测结果应用星期日过滤，并确保预测值不为负"""
        assert self.all_forecast_dates is not None, "请先调用 get_all_forecast_dates()"
        
        print("\n" + "=" * 80)
        print("📅 应用星期日过滤（星期日预测值设为0）")
        print("=" * 80)
        
        # 计算星期日数量
        sunday_count = sum(1 for d in self.all_forecast_dates if d.weekday() == 6)
        workday_count = self.forecast_days - sunday_count
        
        print(f"   - 预测期间: {self.all_forecast_dates[0].strftime('%Y-%m-%d')} ~ {self.all_forecast_dates[-1].strftime('%Y-%m-%d')}")
        print(f"   - 星期日天数: {sunday_count} 天 (预测值设为0)")
        print(f"   - 工作天天数: {workday_count} 天")
        
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
                        # 确保预测值不为负数
                        pred_value = max(0, model['forecast'][workday_idx])
                        filtered_forecast.append(pred_value)
                        workday_idx += 1
                    else:
                        filtered_forecast.append(0)
            
            model['original_forecast'] = model['forecast']
            model['forecast'] = filtered_forecast
        
        print(f"   ✅ 已对 {len(self.all_models)} 个模型应用星期日过滤及负值修正")
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """生成模型对比数据表"""
        assert self.daily_total is not None, "请先调用 filter_data()"
        
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
                '模型名称': model['name'],
                '模型类型': model['model_type'],
                '参数': str(model['params']),
                '历史日均(件)': round(hist_daily, 0),
                '预测日均(含周日)': round(pred_daily, 0),
                '预测工作日日均(件)': round(workday_avg, 0),
                '预测30天总产能(件)': round(pred_total, 0),
                '产能变化率(%)': round(change_pct, 1),
            }
            
            if model.get('evaluation'):
                row['验证MAE'] = round(model['evaluation']['mae'], 0)
                row['验证RMSE'] = round(model['evaluation']['rmse'], 0)
                row['验证MAPE(%)'] = round(model['evaluation']['mape'], 1)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        if '验证MAE' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('验证MAE')
        
        return comparison_df
    
    # ========== 效能评估方法 ==========
    
    def load_actual_production_data(self, production_file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        载入实际生产数据用于评估
        
        Parameters:
        -----------
        production_file_path : str, optional
            实际生产数据Excel文件路径，若为None则使用与训练数据相同的路径
        """
        if production_file_path is None:
            production_file_path = self.file_path.replace('.xlsx', ' - 复制.xlsx')
        
        print("\n" + "=" * 80)
        print("📊 载入实际生产数据进行效能评估")
        print("=" * 80)
        print(f"📂 读取文件: {production_file_path}")
        
        try:
            df_production = pd.read_excel(production_file_path, sheet_name='生产每日交货清单', skipfooter=1)
            df_production["开单日期"] = pd.to_datetime(df_production["开单日期"], format='%Y-%m-%d')
            df_production = df_production.loc[:, ["开单日期", "款号", "发出部门", "交货数"]]
            
            # 筛选车缝部门和指定款号
            df_filtered = df_production[
                df_production['发出部门'].str.contains('車縫', na=False) &
                df_production['款号'].isin(self.items)
            ].copy()
            
            self.actual_production_data = df_filtered.groupby('开单日期')['交货数'].sum().reset_index()
            self.actual_production_data.columns = ['日期', '实际交货数']
            
            print(f"✅ 成功读取 {len(self.actual_production_data)} 笔实际生产数据")
            print(f"   - 日期范围: {self.actual_production_data['日期'].min()} ~ {self.actual_production_data['日期'].max()}")
            
            return self.actual_production_data
            
        except Exception as e:
            print(f"⚠️ 无法载入实际生产数据: {e}")
            print("   将仅使用训练数据中的验证集进行评估")
            self.actual_production_data = None
            return None
    
    def evaluate_with_actual_data(self) -> Optional[pd.DataFrame]:
        """
        使用实际生产数据评估模型预测准确度
        """
        if self.actual_production_data is None:
            print("\n⚠️ 无实际生产数据，跳过效能评估")
            return None
        
        assert self.all_forecast_dates is not None, "请先调用 get_all_forecast_dates()"
        
        print("\n" + "=" * 80)
        print("📈 模型效能评估（使用实际生产数据）")
        print("=" * 80)
        
        # 准备预测数据
        forecast_df = pd.DataFrame({
            '预测日期': self.all_forecast_dates,
            '星期': [d.weekday() for d in self.all_forecast_dates]
        })
        
        for model in self.all_models:
            if model is not None:
                forecast_df[model['name']] = model['forecast']
        
        # 合并预测与实际数据
        merged_df = forecast_df.merge(
            self.actual_production_data,
            left_on='预测日期',
            right_on='日期',
            how='left'
        )
        
        # 只保留有实际观测数据的记录
        valid_df = merged_df[merged_df['实际交货数'].notna() & (merged_df['实际交货数'] > 0)].copy()
        
        if len(valid_df) == 0:
            print("⚠️ 没有可用的实际观测数据进行评估")
            return None
        
        print(f"\n📋 有效观测数据笔数: {len(valid_df)} 笔")
        print(f"   - 日期范围: {valid_df['预测日期'].min()} ~ {valid_df['预测日期'].max()}")
        
        # 计算各模型误差指标
        evaluation_results = []
        
        for model in self.all_models:
            if model is None:
                continue
            y_true = valid_df['实际交货数'].values
            y_pred = valid_df[model['name']].values
            
            # 计算误差指标
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = sqrt(np.mean((y_true - y_pred) ** 2))
            # 避免除以零
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
            
            # R平方
            r2 = r2_score(y_true, y_pred)
            
            # 偏误
            bias = np.mean(y_pred) - np.mean(y_true)
            
            evaluation_results.append({
                '模型名称': model['name'],
                '模型类型': model['model_type'],
                'MAE(件)': round(mae, 0),
                'RMSE(件)': round(rmse, 0),
                'MAPE(%)': round(mape, 1),
                'R²': round(r2, 3),
                '偏误(Bias)': round(bias, 0),
                '平均预测': round(np.mean(y_pred), 0),
                '实际平均': round(np.mean(y_true), 0)
            })
        
        self.evaluation_results = pd.DataFrame(evaluation_results)
        self.evaluation_results = self.evaluation_results.sort_values('MAE(件)')
        
        print("\n🏆 模型效能排名（按MAE由低到高）:")
        print(self.evaluation_results[['模型名称', 'MAE(件)', 'RMSE(件)', 'MAPE(%)', 'R²', '偏误(Bias)']].head(15).to_string(index=False))
        
        return self.evaluation_results
    
    def categorize_evaluation_models(self) -> Optional[pd.DataFrame]:
        """对评估结果进行模型分类统计"""
        if self.evaluation_results is None:
            return None
        
        print("\n" + "=" * 80)
        print("📊 各类型模型效能对比")
        print("=" * 80)
        
        model_categories = {
            '移动平均': ['SMA-3', 'SMA-7', 'SMA-15', 'SMA-30'],
            '加权移动平均': ['WMA-7', 'WMA-15'],
            '指数平滑': ['ES-α=0.3', 'ES-α=0.5', 'ES-α=0.7'],
            '双指数平滑': ['Holt-α=0.5,β=0.3', 'Holt-α=0.6,β=0.2'],
            '季节性预测': ['Seasonal-周期7天', 'Seasonal-周期15天', 'Seasonal-周期30天'],
            '线性回归': ['Linear_Regression'],
            '多项式回归': ['Poly2_Regression', 'Poly3_Regression'],
            '正则化回归': ['Ridge_Regression', 'Ridge_α10_Regression', 'Lasso_Regression'],
            '自回归': ['AR-3', 'AR-7', 'AR-14'],
            '移动平均回归': ['MA_Trend-7', 'MA_Trend-14']
        }
        
        category_results = []
        for category, models in model_categories.items():
            category_models = self.evaluation_results[self.evaluation_results['模型名称'].isin(models)]
            if len(category_models) > 0:
                category_results.append({
                    '模型类型': category,
                    '平均MAE': round(category_models['MAE(件)'].mean(), 0),
                    '平均MAPE(%)': round(category_models['MAPE(%)'].mean(), 1),
                    '最佳模型': category_models.iloc[category_models['MAE(件)'].argmin()]['模型名称'],
                    '最佳MAE': category_models['MAE(件)'].min()
                })
        
        self.category_evaluation = pd.DataFrame(category_results)
        self.category_evaluation = self.category_evaluation.sort_values('平均MAE')
        print(self.category_evaluation.to_string(index=False))
        
        return self.category_evaluation
    
    def plot_evaluation_charts(self) -> None:
        """生成效能评估图表"""
        if self.evaluation_results is None or len(self.evaluation_results) == 0:
            print("⚠️ 无评估结果，跳过图表生成")
            return
        
        best_model_row = self.evaluation_results.iloc[0]
        best_model_name = best_model_row['模型名称']
        best_mae = best_model_row['MAE(件)']
        best_mape = best_model_row['MAPE(%)']
        best_r2 = best_model_row['R²']
        
        # 准备数据
        assert self.all_forecast_dates is not None, "请先调用 get_all_forecast_dates()"
        assert self.actual_production_data is not None, "请先调用 load_actual_production_data()"
        
        forecast_df = pd.DataFrame({'预测日期': self.all_forecast_dates})
        for model in self.all_models:
            if model is not None:
                forecast_df[model['name']] = model['forecast']
        
        merged_df = forecast_df.merge(
            self.actual_production_data,
            left_on='预测日期',
            right_on='日期',
            how='left'
        )
        valid_df = merged_df[merged_df['实际交货数'].notna() & (merged_df['实际交货数'] > 0)].copy()
        
        if len(valid_df) == 0:
            print("⚠️ 无有效数据生成图表")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'车缝组 - {self.items[:3] if len(self.items) > 3 else self.items}...预测模型效能评估 (最佳模型: {best_model_name})', fontsize=14, fontweight='bold')
        
        # 图1: 最佳模型预测 vs 实际值
        ax = axes[0, 0]
        df_sorted = valid_df.sort_values('预测日期')
        ax.plot(df_sorted['预测日期'], df_sorted['实际交货数'], 'b-o', linewidth=2, markersize=6, label='实际观测值')
        ax.plot(df_sorted['预测日期'], df_sorted[best_model_name], 'r--s', linewidth=2, markersize=6, label=f'{best_model_name} 预测值')
        ax.set_title(f'最佳模型: {best_model_name}\nMAE={best_mae:.0f}, MAPE={best_mape:.1f}%, R²={best_r2:.3f}', 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('交货数 (件)')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 图2: 误差柱状图
        ax = axes[0, 1]
        top_models = self.evaluation_results.head(10)
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(top_models))]
        bars = ax.barh(top_models['模型名称'], top_models['MAE(件)'], color=colors, edgecolor='black')
        ax.set_title('模型误差对比 (MAE - 越低越好)', fontsize=11, fontweight='bold')
        ax.set_xlabel('平均绝对误差 (件)')
        ax.axvline(x=top_models['MAE(件)'].mean(), color='red', linestyle='--', alpha=0.7, 
                   label=f'平均MAE: {top_models["MAE(件)"].mean():.0f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, top_models['MAE(件)']):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, f'{val:.0f}', 
                   va='center', fontsize=9)
        
        # 图3: 预测误差分布
        ax = axes[1, 0]
        errors = valid_df['实际交货数'].values - valid_df[best_model_name].values
        ax.hist(errors, bins=15, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
        ax.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
                   label=f'平均误差: {np.mean(errors):.0f}')
        ax.set_title(f'{best_model_name} 预测误差分布', fontsize=11, fontweight='bold')
        ax.set_xlabel('预测误差 (件)')
        ax.set_ylabel('频次')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 图4: 散点图
        ax = axes[1, 1]
        ax.scatter(valid_df['实际交货数'], valid_df[best_model_name], alpha=0.7, s=80, 
                   c='steelblue', edgecolors='navy')
        max_val = max(valid_df['实际交货数'].max(), valid_df[best_model_name].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美预测线')
        ax.set_title(f'{best_model_name} - 预测值 vs 实际值', fontsize=11, fontweight='bold')
        ax.set_xlabel('实际交货数 (件)')
        ax.set_ylabel('预测交货数 (件)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '模型效能评估对比图.png'), dpi=150, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ 效能评估图表已储存")
    
    def plot_all_models_comparison(self) -> None:
        """生成所有模型预测对比图"""
        if self.actual_production_data is None:
            return
        
        assert self.all_forecast_dates is not None, "请先调用 get_all_forecast_dates()"
        
        forecast_df = pd.DataFrame({'预测日期': self.all_forecast_dates})
        for model in self.all_models:
            if model is not None:
                forecast_df[model['name']] = model['forecast']
        
        merged_df = forecast_df.merge(
            self.actual_production_data,
            left_on='预测日期',
            right_on='日期',
            how='left'
        )
        valid_df = merged_df[merged_df['实际交货数'].notna() & (merged_df['实际交货数'] > 0)].copy()
        
        if len(valid_df) == 0:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('车缝组 - 所有模型预测 vs 实际对比', fontsize=14, fontweight='bold')
        
        for idx, model in enumerate(self.all_models[:6]):
            if model is None:
                continue
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            df_sorted = valid_df.sort_values('预测日期')
            ax.plot(df_sorted['预测日期'], df_sorted['实际交货数'], 'b-o', linewidth=2, markersize=5, label='实际观测值')
            ax.plot(df_sorted['预测日期'], df_sorted[model['name']], 'r--s', linewidth=2, markersize=5, label=f'{model["name"]}')
            
            assert self.evaluation_results is not None
            model_mae = self.evaluation_results[self.evaluation_results['模型名称'] == model['name']]['MAE(件)'].values[0]
            ax.set_title(f'{model["name"]} (MAE: {model_mae:.0f}件)', fontsize=10, fontweight='bold')
            ax.set_xlabel('日期')
            ax.set_ylabel('交货数 (件)')
            ax.legend(fontsize=8)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '所有模型预测对比图.png'), dpi=150, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ 所有模型对比图已储存")
    
    def print_evaluation_recommendations(self) -> None:
        """打印模型推荐建议"""
        if self.evaluation_results is None or len(self.evaluation_results) == 0:
            return
        
        best_model = self.evaluation_results.iloc[0]['模型名称']
        best_mae = self.evaluation_results.iloc[0]['MAE(件)']
        best_mape = self.evaluation_results.iloc[0]['MAPE(%)']
        second_best = self.evaluation_results.iloc[1]['模型名称'] if len(self.evaluation_results) > 1 else None
        
        print("\n" + "=" * 80)
        print("💡 模型推荐建议")
        print("=" * 80)
        
        print(f"""
📌 最佳模型: {best_model}
   - 优点: 预测误差最小，准确度最高
   - 适用场景: 一般生产规划与预测
   - MAE: {best_mae:.0f}件 | MAPE: {best_mape:.1f}%

📌 次佳模型: {second_best}
   - 可作为备用方案或模型集成使用

📌 各类型模型推荐:
""")
        
        if self.category_evaluation is not None:
            for _, row in self.category_evaluation.iterrows():
                print(f"   • {row['模型类型']}: {row['最佳模型']} (MAE: {row['最佳MAE']:.0f}件)")
        
        # 找出SMA模型
        sma_models = self.evaluation_results[self.evaluation_results['模型名称'].str.contains('SMA')]
        best_sma = sma_models.iloc[0]['模型名称'] if len(sma_models) > 0 else 'SMA模型'
        
        # 找出季节性模型
        seasonal_models = self.category_evaluation[self.category_evaluation['模型类型'] == '季节性预测'] if self.category_evaluation is not None and len(self.category_evaluation) > 0 else pd.DataFrame()
        best_seasonal = seasonal_models['最佳模型'].values[0] if len(seasonal_models) > 0 else '季节性模型'
        
        # 找出回归模型
        regression_models = self.evaluation_results[self.evaluation_results['模型名称'].str.contains('Reg|AR-')]
        best_regression = regression_models.iloc[0]['模型名称'] if len(regression_models) > 0 else '回归模型'
        
        print(f"""
📊 模型选择建议:
   1. 优先使用 {best_model} 进行预测
   2. 若需要预测稳定性，可考虑 {best_sma}
   3. 若数据有季节性趋势，可参考 {best_seasonal}
   4. 若需要捕捉长期趋势，可考虑回归模型: {best_regression}
   5. 建议使用多模型集成（取平均）来降低单一模型风险
""")
    
    def save_evaluation_excel(self) -> None:
        """保存效能评估Excel报告"""
        if self.evaluation_results is None:
            return
        
        excel_file = os.path.join(self.output_folder, '模型效能评估报告.xlsx')
        
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 模型排名表
                self.evaluation_results.to_excel(writer, sheet_name='模型排名', index=False)
                
                # 各模型类型汇总
                if self.category_evaluation is not None:
                    self.category_evaluation.to_excel(writer, sheet_name='模型类型汇总', index=False)
                
                # 预测vs实际明细
                assert self.all_forecast_dates is not None
                forecast_df = pd.DataFrame({'预测日期': self.all_forecast_dates})
                for model in self.all_models:
                    if model is not None:
                        forecast_df[model['name']] = model['forecast']
                
                assert self.actual_production_data is not None
                merged_df = forecast_df.merge(
                    self.actual_production_data,
                    left_on='预测日期',
                    right_on='日期',
                    how='left'
                )
                detail_df = merged_df[merged_df['实际交货数'].notna() & (merged_df['实际交货数'] > 0)].copy()
                detail_df['星期'] = detail_df['预测日期'].dt.dayofweek
                detail_df.to_excel(writer, sheet_name='预测vs实际明细', index=False)
                
                # 模型推荐
                best_model = self.evaluation_results.iloc[0]['模型名称']
                second_best = self.evaluation_results.iloc[1]['模型名称'] if len(self.evaluation_results) > 1 else 'N/A'
                sma_models = self.evaluation_results[self.evaluation_results['模型名称'].str.contains('SMA')]
                best_sma = sma_models.iloc[0]['模型名称'] if len(sma_models) > 0 else 'SMA-7'
                
                recommendation = pd.DataFrame({
                    '推荐类型': ['最佳准确度', '备用方案', '最稳定(移动平均)', '保守预测', '乐观预测'],
                    '推荐模型': [
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
                recommendation.to_excel(writer, sheet_name='模型推荐', index=False)
            
            print(f"\n✅ Excel评估报告已储存: {excel_file}")
        except Exception as e:
            print(f"\n⚠️ Excel储存失败: {e}")
    
    # ========== 图表生成方法（原有功能） ==========
    
    def plot_comparison_charts(self) -> Optional[Dict[str, Any]]:
        """生成多模型对比图表"""
        print("\n" + "=" * 80)
        print("📈 生成多模型对比图表")
        print("=" * 80)
        
        # 过滤掉 None 模型
        valid_models = [m for m in self.all_models if m is not None]
        if not valid_models:
            print("⚠️ 没有有效的模型进行对比")
            return None
        
        assert self.daily_total is not None, "请先调用 filter_data()"
        assert self.all_forecast_dates is not None, "请先调用 get_all_forecast_dates()"
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'车缝组{self.items[:3] if len(self.items) > 3 else self.items}* - 多模型预测对比分析 (排除星期日)', fontsize=16, fontweight='bold')
        
        # 安全获取 MAE 的辅助函数
        def get_model_mae(model: Dict[str, Any]) -> float:
            evaluation = model.get('evaluation')
            if evaluation and isinstance(evaluation, dict):
                return evaluation.get('mae', float('inf'))
            return float('inf')
        
        best_model = min(valid_models, key=get_model_mae)
        
        # 1. 历史数据 + 最佳模型预测
        ax = axes[0, 0]
        ax.plot(self.daily_total.index[-60:], self.daily_total.values[-60:], 
                'b-o', linewidth=1.5, markersize=3, label='历史实际交货')
        
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
                      label='星期日(不生产)')
        
        ax.axhline(y=self.daily_total.mean(), color='blue', linestyle=':', alpha=0.5, 
                   label=f'历史日均: {self.daily_total.mean():.0f}')
        ax.axhline(y=sum(best_model['forecast'])/self.forecast_days, color='red', linestyle=':', alpha=0.5, 
                   label=f'预测日均(含周日): {sum(best_model["forecast"])/self.forecast_days:.0f}')
        
        # 安全获取评价值
        best_eval = best_model.get('evaluation')
        best_mae = best_eval.get('mae', 'N/A') if best_eval else 'N/A'
        best_mape = best_eval.get('mape', 'N/A') if best_eval else 'N/A'
        
        title_text = f'最佳模型预测 - {best_model["name"]}'
        if best_mae != 'N/A' and best_mape != 'N/A':
            title_text += f'\nMAE: {best_mae:.0f} | MAPE: {best_mape:.1f}%'
        ax.set_title(title_text, fontsize=11, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('交货数 (件)')
        ax.legend(loc='upper left', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 2. 模型误差对比
        ax = axes[0, 1]
        models_with_mae = [(m['name'], m.get('evaluation', {}).get('mae', 0) if m.get('evaluation') else 0) 
                          for m in valid_models]
        models_with_mae = [(name, mae) for name, mae in models_with_mae if mae > 0]  # 只保留有评估值的模型
        models_with_mae.sort(key=lambda x: x[1])
        if models_with_mae:
            names = [m[0][:15] for m in models_with_mae[:10]]
            maes = [m[1] for m in models_with_mae[:10]]
            bars = ax.barh(names, maes, color='skyblue', edgecolor='navy')
            ax.set_title('模型误差对比 (MAE - 越低越好)', fontsize=11, fontweight='bold')
            ax.set_xlabel('平均绝对误差 (件)')
            ax.grid(True, alpha=0.3, axis='x')
            for bar, val in zip(bars, maes):
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f'{val:.0f}', 
                       va='center', fontsize=9)
        
        # 3. 模型预测范围对比
        ax = axes[0, 2]
        all_predictions = [m['forecast'] for m in valid_models]
        pred_array = np.array(all_predictions)
        pred_mean = pred_array.mean(axis=0)
        pred_min = pred_array.min(axis=0)
        pred_max = pred_array.max(axis=0)
        
        ax.fill_between(range(self.forecast_days), pred_min, pred_max, alpha=0.3, color='gray', 
                       label='预测范围(最小-最大)')
        ax.plot(range(self.forecast_days), pred_mean, 'b-', linewidth=2, label='平均预测')
        ax.plot(range(self.forecast_days), [self.daily_total.mean()] * self.forecast_days, 'r--', 
               label='历史日均线')
        ax.set_title(f'多模型预测范围 (共{len(valid_models)}个模型, 排除星期日)', fontsize=11, fontweight='bold')
        ax.set_xlabel('预测天数')
        ax.set_ylabel('预测交货数 (件)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 最佳3个模型预测对比
        ax = axes[1, 0]
        top_3 = sorted(valid_models, key=get_model_mae)[:3]
        colors = ['red', 'green', 'orange']
        for i, model in enumerate(top_3):
            non_sunday_vals = [model['forecast'][j] for j in range(self.forecast_days) 
                              if self.all_forecast_dates[j].weekday() != 6]
            if non_sunday_dates:
                # 安全获取模型MAE
                model_eval = model.get('evaluation')
                model_mae = model_eval.get('mae', 0) if model_eval else 0
                ax.plot(non_sunday_dates, non_sunday_vals, color=colors[i], linewidth=2, 
                        label=f"{model['name']} (MAE:{model_mae:.0f})")
        ax.plot(self.daily_total.index[-30:], self.daily_total.values[-30:], 'b-o', linewidth=1.5, markersize=3, 
               label='历史数据')
        ax.set_title('最佳3个模型预测对比', fontsize=11, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('交货数 (件)')
        ax.legend(loc='upper left', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 5. 模型类型性能对比
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
            ax.set_title('各类型模型平均性能对比', fontsize=11, fontweight='bold')
            ax.set_ylabel('平均MAE (件)')
            ax.tick_params(axis='x', rotation=15)
            ax.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars, type_mae_means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', 
                       ha='center', fontsize=9)
        
        # 6. 星期日影响分析
        ax = axes[1, 2]
        assert self.df_filtered is not None
        daily_total_all_with_weekday = self.df_filtered.copy()
        daily_total_all_with_weekday['星期'] = daily_total_all_with_weekday['开单日期'].dt.dayofweek
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        
        # 计算各星期平均交货量
        weekday_avg = daily_total_all_with_weekday.groupby('星期')['交货数'].mean()
        
        # 确保所有星期都有数据（补齐缺少的星期）
        for i in range(7):
            if i not in weekday_avg.index:
                weekday_avg[i] = 0
        
        # 重新排序并设置索引名称
        weekday_avg = weekday_avg.reindex(range(7))
        weekday_avg.index = weekday_names
        
        # 设置颜色（星期日用不同颜色）
        colors_bar = ['steelblue'] * 6 + ['lightcoral']
        bars = ax.bar(weekday_names, weekday_avg.values, color=colors_bar, edgecolor='black')
        ax.set_title('各星期平均交货量分析', fontsize=11, fontweight='bold')
        ax.set_xlabel('星期')
        ax.set_ylabel('平均交货数 (件)')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, weekday_avg.values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', 
                       ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '00_多模型预测对比总图.png'), dpi=150, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ 多模型对比总图已储存")
        
        return best_model
    
    def plot_detailed_forecasts(self, max_models: int = 6) -> None:
        """生成各模型详细预测图"""
        print("\n📊 生成各模型详细预测图...")
        
        valid_models = [m for m in self.all_models if m is not None]
        
        assert self.daily_total is not None, "请先调用 filter_data()"
        assert self.all_forecast_dates is not None, "请先调用 get_all_forecast_dates()"
        
        for i, model in enumerate(valid_models[:max_models]):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            ax = axes[0]
            show_days = min(60, len(self.daily_total))
            ax.plot(self.daily_total.index[-show_days:], self.daily_total.values[-show_days:], 
                    'b-o', linewidth=1.5, markersize=3, label='历史交货量')
            
            non_sunday_dates = [self.all_forecast_dates[j] for j in range(self.forecast_days) 
                               if self.all_forecast_dates[j].weekday() != 6]
            non_sunday_vals = [model['forecast'][j] for j in range(self.forecast_days) 
                              if self.all_forecast_dates[j].weekday() != 6]
            sunday_dates = [self.all_forecast_dates[j] for j in range(self.forecast_days) 
                           if self.all_forecast_dates[j].weekday() == 6]
            
            if non_sunday_dates:
                ax.plot(non_sunday_dates, non_sunday_vals, 'r--o', linewidth=2, markersize=4, label='预测值')
            if sunday_dates:
                ax.scatter(sunday_dates, [0]*len(sunday_dates), color='gray', marker='x', s=50, 
                          label='星期日(不生产)')
            
            hist_std = self.daily_total.std()
            upper = [model['forecast'][j] + hist_std for j in range(self.forecast_days)]
            lower = [max(0, model['forecast'][j] - hist_std) for j in range(self.forecast_days)]
            ax.fill_between(self.all_forecast_dates, lower, upper, alpha=0.2, color='red', label='±1σ 置信区间')
            
            ax.axhline(y=self.daily_total.mean(), color='blue', linestyle=':', alpha=0.5, 
                      label=f'历史日均: {self.daily_total.mean():.0f}')
            ax.axhline(y=sum(model['forecast'])/self.forecast_days, color='red', linestyle=':', alpha=0.5, 
                      label=f'预测日均(含周日): {sum(model["forecast"])/self.forecast_days:.0f}')
            
            workday_avg = np.mean(non_sunday_vals) if non_sunday_vals else 0
            ax.axhline(y=workday_avg, color='orange', linestyle=':', alpha=0.5, 
                      label=f'预测工作日日均: {workday_avg:.0f}')
            
            ax.set_title(f'{model["name"]} - {model["model_type"]}\n预测30天总产能: {sum(model["forecast"]):,.0f}件 (星期日不生产)', 
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('日期')
            ax.set_ylabel('交货数 (件)')
            ax.legend(loc='upper left', fontsize=8)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            ax = axes[1]
            if model.get('evaluation'):
                ev = model['evaluation']
                days = list(range(1, len(ev['predictions']) + 1))
                ax.plot(days, ev['actual'], 'b-o', linewidth=2, markersize=6, label='实际值')
                ax.plot(days, ev['predictions'], 'r--s', linewidth=2, markersize=6, label='预测值')
                
                errors = np.abs(ev['actual'] - ev['predictions'])
                for d, a, p, e in zip(days, ev['actual'], ev['predictions'], errors):
                    ax.vlines(d, min(a, p), max(a, p), color='gray', alpha=0.5, linewidth=1)
                
                ax.set_title(f'验证集表现 (最近{self.test_days}个工作日)\nMAE: {ev["mae"]:.0f} | RMSE: {ev["rmse"]:.0f} | MAPE: {ev["mape"]:.1f}%', 
                             fontsize=12, fontweight='bold')
                ax.set_xlabel('验证天数')
                ax.set_ylabel('交货数 (件)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                if non_sunday_vals:
                    ax.hist(non_sunday_vals, bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
                    ax.axvline(x=np.mean(non_sunday_vals), color='red', linestyle='--', linewidth=2, 
                              label=f'工作日均值: {np.mean(non_sunday_vals):.0f}')
                ax.set_title(f'预测值分布 (仅工作日)', fontsize=12, fontweight='bold')
                ax.set_xlabel('预测交货数 (件)')
                ax.set_ylabel('频次')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_name = model['name'].replace('=', '').replace(',', '_').replace('-', '_')
            plt.savefig(os.path.join(self.output_folder, f'{safe_name}_详细预测.png'), dpi=150, 
                       bbox_inches='tight', facecolor='white')
            plt.close()
        
        print(f"   ✅ 已生成 {min(max_models, len(valid_models))} 个详细预测图")
    
    def save_excel_report(
        self, 
        best_model: Dict[str, Any], 
        comparison_df: pd.DataFrame, 
        pred_mean: np.ndarray, 
        pred_min: np.ndarray, 
        pred_max: np.ndarray
    ) -> None:
        """保存Excel综合报告"""
        print("\n📊 生成Excel综合报告...")
        
        excel_file = os.path.join(self.output_folder, '00_多模型预测综合报告.xlsx')
        
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 原始资料
                assert self.df_filtered is not None
                self.df_filtered.to_excel(writer, sheet_name='原始资料', index=False)
                pd.DataFrame(self.items, columns=['款号']).to_excel(writer, sheet_name='观测款号', index=False)
                comparison_df.to_excel(writer, sheet_name='模型对比总表', index=False)
                
                assert self.all_forecast_dates is not None
                best_model_forecast = pd.DataFrame({
                    '预测日期': self.all_forecast_dates,
                    '星期': [['周一', '周二', '周三', '周四', '周五', '周六', '周日'][d.weekday()] for d in self.all_forecast_dates],
                    '预测交货数': [round(v) for v in best_model['forecast']],
                    '下限(±1σ)': [max(0, round(v - self.daily_total.std())) for v in best_model['forecast']],
                    '上限(±1σ)': [round(v + self.daily_total.std()) for v in best_model['forecast']]
                })
                best_model_forecast.to_excel(writer, sheet_name=f'最佳模型_{best_model["name"]}_预测', index=False)
                
                all_forecasts = pd.DataFrame({
                    '预测日期': self.all_forecast_dates,
                    '星期': [['周一', '周二', '周三', '周四', '周五', '周六', '周日'][d.weekday()] for d in self.all_forecast_dates]
                })
                for model in self.all_models:
                    if model is not None:
                        all_forecasts[model['name']] = [round(v) for v in model['forecast']]
                all_forecasts.to_excel(writer, sheet_name='所有模型预测汇总', index=False)
                
                sunday_count = sum(1 for d in self.all_forecast_dates if d.weekday() == 6)
                
                # 安全获取评价值
                best_eval = best_model.get('evaluation')
                best_mae_str = f"{best_eval['mae']:.0f}" if best_eval else 'N/A'
                best_mape_str = f"{best_eval['mape']:.1f}%" if best_eval else 'N/A'
                
                summary = pd.DataFrame({
                    '统计项目': [
                        '历史总交货量(件)',
                        '历史有交货天数(总)',
                        '历史有交货天数(排除周日)',
                        '历史日均产能(件/天)',
                        '历史标准差(件)',
                        '评估模型数量',
                        '最佳模型(按MAE)',
                        '最佳模型MAE(件)',
                        '最佳模型MAPE(%)',
                        '多模型平均预测日均(件/天)',
                        '多模型预测范围(件/天)',
                        '星期日处理方式'
                    ],
                    '数值': [
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
                        '星期日预测值设为0（不生产）'
                    ]
                })
                summary.to_excel(writer, sheet_name='统计摘要', index=False)
                
                valid_models = [m for m in self.all_models if m is not None]
                recommendation = pd.DataFrame({
                    '使用场景': ['短期波动大', '一般生产规划', '平稳生产', '保守预测', '乐观预测'],
                    '推荐模型': [
                        'SMA-3 或 ES-α=0.7',
                        'SMA-7 或 WMA-7',
                        'SMA-15 或 ES-α=0.3',
                        f'{min(valid_models, key=lambda x: sum(x["forecast"]))["name"]}' if valid_models else 'N/A',
                        f'{max(valid_models, key=lambda x: sum(x["forecast"]))["name"]}' if valid_models else 'N/A'
                    ],
                    '预测工作日日均(件)': [
                        'N/A', 'N/A', 'N/A',
                        f"{sum(min(valid_models, key=lambda x: sum(x['forecast']))['forecast'])/self.forecast_days:.0f}" if valid_models else 'N/A',
                        f"{sum(max(valid_models, key=lambda x: sum(x['forecast']))['forecast'])/self.forecast_days:.0f}" if valid_models else 'N/A'
                    ]
                })
                recommendation.to_excel(writer, sheet_name='模型推荐', index=False)
            
            print(f"   ✅ Excel报告已储存: {excel_file}")
        except Exception as e:
            print(f"   ⚠️ Excel储存失败: {e}")
    
    def print_final_report(
        self, 
        best_model: Dict[str, Any], 
        pred_mean: np.ndarray, 
        pred_min: np.ndarray, 
        pred_max: np.ndarray, 
        sunday_count: int
    ) -> None:
        """打印最终报告"""
        print("\n" + "=" * 80)
        print("📋 多模型预测最终报告 (排除星期日)")
        print("=" * 80)
        
        print(f"\n🏆 最佳模型: {best_model['name']}")
        best_eval = best_model.get('evaluation')
        if best_eval:
            print(f"   - MAE (平均绝对误差): {best_eval['mae']:.0f} 件")
            print(f"   - RMSE (均方根误差): {best_eval['rmse']:.0f} 件")
            print(f"   - MAPE (平均绝对百分比误差): {best_eval['mape']:.1f}%")
        print(f"   - 预测30天总产能: {sum(best_model['forecast']):,.0f} 件")
        print(f"   - 预测日均产能(含周日): {sum(best_model['forecast'])/self.forecast_days:.0f} 件/天")
        workday_forecast = [v for v in best_model['forecast'] if v > 0]
        if workday_forecast:
            print(f"   - 预测工作日日均产能: {np.mean(workday_forecast):.0f} 件/天")
        
        print(f"\n📊 多模型统计:")
        print(f"   - 模型总数: {len([m for m in self.all_models if m is not None])}")
        print(f"   - 预测日均范围(含周日): {pred_min.mean():.0f} ~ {pred_max.mean():.0f} 件/天")
        print(f"   - 预测30天总产能范围: {pred_min.sum():.0f} ~ {pred_max.sum():.0f} 件")
        
        print(f"\n📅 星期日处理说明:")
        print(f"   - 历史数据训练时已排除星期日")
        print(f"   - 未来预测中星期日统一设为0")
        print(f"   - 共排除 {len(self.daily_total_all) - len(self.daily_total)} 天历史星期日数据")
        print(f"   - 预测期间共有 {sunday_count} 个星期日")
        
        print(f"\n📁 输出档案结构:")
        print(f"\n{self.output_folder}/")
        print("   ├── 00_多模型预测综合报告.xlsx          (所有模型对比)")
        print("   ├── 00_多模型预测对比总图.png          (6合1对比图)")
        print("   ├── [模型名称]_详细预测.png            (各模型详细图)")
        print("   ├── 模型效能评估报告.xlsx              (效能评估报告)")
        print("   ├── 模型效能评估对比图.png              (效能评估图表)")
        print("   ├── 所有模型预测对比图.png              (模型对比图)")
        print("   └── ...")
        
        print("\n" + "=" * 80)
        print("⚠️ 重要说明:")
        print(f"   1. 本报告使用{self.test_days}天历史数据进行模型验证")
        print("   2. 星期日不生产，预测值设为0")
        print("   3. 历史训练数据已排除星期日")
        print("   4. MAE越低代表模型越准确")
        print("   5. 预测工作日日均更能反映实际产能")
        print("   6. 所有预测值均已确保不为负数（Holt模型负值已修正）")
        print("   7. 新增回归分析模型（线性、多项式、岭回归、自回归等）")
        print("=" * 80)
        
        print("\n✨ 多模型预测对比报告生成完成！")
        print(f"📁 报告已储存至: {os.path.abspath(self.output_folder)}")
    
    # ========== 主运行方法 ==========
    
    def run(self, production_file_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        执行完整的预测流程
        
        Parameters:
        -----------
        production_file_path : str, optional
            实际生产数据文件路径，用于效能评估
        """
        # 1. 加载数据
        self.load_data()
        
        # 2. 筛选数据
        self.filter_data()
        
        # 3. 获取预测日期
        self.get_all_forecast_dates()
        
        # 4. 训练所有模型
        self.train_models()
        
        # 5. 应用星期日过滤
        self.apply_sunday_filter()
        
        # 6. 获取对比数据表
        comparison_df = self.get_comparison_dataframe()
        
        # 7. 生成对比图表
        best_model = self.plot_comparison_charts()
        
        # 8. 生成详细预测图
        self.plot_detailed_forecasts()
        
        # 9. 计算统计数据
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
        
        # 10. 保存Excel报告
        if best_model is not None:
            self.save_excel_report(best_model, comparison_df, pred_mean, pred_min, pred_max)
        
        # 11. 打印最终报告
        if best_model is not None:
            self.print_final_report(best_model, pred_mean, pred_min, pred_max, sunday_count)
        else:
            print("\n⚠️ 无法生成最终报告：没有有效的模型")
        
        # 12. 效能评估（如果有实际数据）
        if production_file_path is not None or os.path.exists(self.file_path.replace('.xlsx', ' - 复制.xlsx')):
            try:
                self.load_actual_production_data(production_file_path)
                self.evaluate_with_actual_data()
                self.categorize_evaluation_models()
                self.plot_evaluation_charts()
                self.plot_all_models_comparison()
                self.print_evaluation_recommendations()
                self.save_evaluation_excel()
            except Exception as e:
                print(f"⚠️ 效能评估失败: {e}")
                print("   将跳过效能评估部分")
        else:
            print("\n⚠️ 未找到实际生产数据文件，跳过效能评估")
            print("   如需进行效能评估，请准备文件名为 '生产每日交货清单 - 复制.xlsx' 的实际数据文件")
        
        return best_model


def get_防火料款號() -> List[str]:
    wb_path = r"C:\Users\Dell\Desktop\git_demo\防火料款號.xlsx"
    sht = '表格资料'
    df = pd.read_excel(wb_path, sheet_name=sht)
    return df['item'].to_list()


# ============================================
# 使用范例
# ============================================
if __name__ == "__main__":
    # 配置参数
    # items = get_防火料款號()
    # items = ['P11774-8', 'P11774-7', 'P11775-7', 'P11775-8']  # 可以修改为需要的款号
    items = ['P12857-1','P12857-4', 'P12857-TD6', 'P12857-3A', 'P12857-TM10']
    items = ['P13040-3', 'P13040-1']
    file_path = r'C:\Users\Dell\Downloads\生产每日交货清单.xlsx'
    production_file_path = r'C:\Users\Dell\Downloads\生产每日交货清单(实际生产数据).xlsx'
    
    # 创建分析器实例并运行
    analyzer = SewingForecastAnalyzer(
        file_path=file_path,
        items=items,
        test_days=60,
        forecast_days=30,
        output_folder='车缝组_多模型预测对比报告(排除周日)'
    )
    
    # 执行完整分析（包含效能评估）
    best_model = analyzer.run(production_file_path)
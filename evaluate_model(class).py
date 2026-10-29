import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import warnings
import duckdb as dd
import os
from datetime import datetime

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    預測模型效能評估器
    用於評估多個預測模型在實際觀測數據上的表現
    """
    
    def __init__(self, forecast_excel_path, forecast_sheet_name, 
                 production_excel_path, production_sheet_name,
                 items=None, output_folder=None):
        """
        初始化模型評估器
        
        Parameters:
        -----------
        forecast_excel_path : str
            預測報告Excel文件路徑
        forecast_sheet_name : str
            預測報告工作表名稱
        production_excel_path : str
            生產數據Excel文件路徑
        production_sheet_name : str
            生產數據工作表名稱
        items : list, optional
            款號列表，默認為P12857相關款號
        output_folder : str, optional
            輸出資料夾名稱
        """
        self.forecast_excel_path = forecast_excel_path
        self.forecast_sheet_name = forecast_sheet_name
        self.production_excel_path = production_excel_path
        self.production_sheet_name = production_sheet_name
        
        # 設置款號
        if items is None:
            self.items = ['P1174-7', 'P11774-8', 'P11775-7', 'P11775-8', 'P11776-7',
                         'P11776-8', 'P12092-4', 'P12092-5', 'P12093-5', 'P12093-4',
                         'P12094-4', 'P12094-5', 'P12297-5', 'P12297-4']
        else:
            self.items = items
        
        # 輸出設置
        if output_folder is None:
            self.output_folder = '模型效能評估報告'
        else:
            self.output_folder = output_folder
        
        # 數據屬性
        self.df_forecast = None
        self.df_production = None
        self.df_items = None
        self.df_merged = None
        self.df_valid = None
        self.model_columns = []
        self.results_df = None
        self.category_df = None
        
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
        """加載預測數據和生產數據"""
        print("=" * 100)
        print("📊 車縫組 - 預測模型效能評估分析報告")
        print("=" * 100)
        
        # 讀取預測數據
        print(f"\n📂 讀取預測報告: {self.forecast_excel_path}")
        self.df_forecast = pd.read_excel(self.forecast_excel_path, sheet_name=self.forecast_sheet_name)
        print(f"   ✅ 讀取 {len(self.df_forecast)} 筆預測數據")
        
        # 創建款號DataFrame
        self.df_items = pd.DataFrame(self.items, columns=['item'])
        
        # 讀取生產數據
        print(f"📂 讀取生產數據: {self.production_excel_path}")
        self.df_production = pd.read_excel(self.production_excel_path, sheet_name=self.production_sheet_name, skipfooter=1)
        self.df_production["开单日期"] = pd.to_datetime(self.df_production["开单日期"], format='%Y-%m-%d')
        self.df_production = self.df_production.loc[:, ["开单日期", "款号", "发出部门", "交货数"]]
        print(f"   ✅ 讀取 {len(self.df_production)} 筆生產數據")
        
        return self
    
    def filter_and_merge(self):
        """篩選車縫部門數據並與預測數據合併"""
        print("\n🔍 篩選車縫部門數據...")
        
        # 使用DuckDB查詢車縫部門數據
        sql = f"""
            SELECT 开单日期, SUM(交货数) as 交货数 
            FROM self.df_production 
            WHERE 发出部门 LIKE '%車縫%' 
            AND 款号 IN (SELECT item FROM self.df_items) 
            GROUP BY 开单日期
        """
        
        # 由於DuckDB無法直接訪問類屬性，這裡改用pandas方式
        df_filtered = self.df_production[
            self.df_production['发出部门'].str.contains('車縫', na=False) &
            self.df_production['款号'].isin(self.items)
        ].copy()
        
        df_production_sum = df_filtered.groupby('开单日期')['交货数'].sum().reset_index()
        df_production_sum.columns = ['开单日期', '交货数']
        
        print(f"   ✅ 篩選後 {len(df_production_sum)} 筆生產數據")
        
        # 合併預測與實際數據
        print("📊 合併預測與實際數據...")
        self.df_merged = self.df_forecast.merge(
            df_production_sum, 
            left_on='預測日期', 
            right_on='开单日期', 
            how='left'
        )
        self.df_merged.rename(columns={'交货数': '觀測數據'}, inplace=True)
        
        # 保存原始合併數據
        self.df_merged.to_excel(os.path.join(self.output_folder, '觀測報告.xlsx'), index=False)
        print(f"   ✅ 合併完成，共 {len(self.df_merged)} 筆記錄")
        
        return self
    
    def clean_data(self):
        """清理數據，只保留有觀測數據的記錄"""
        self.df_valid = self.df_merged[
            self.df_merged['觀測數據'].notna() & (self.df_merged['觀測數據'] > 0)
        ].copy()
        
        print(f"\n📋 有效觀測數據筆數: {len(self.df_valid)} 筆")
        print(f"   - 日期範圍: {self.df_valid['預測日期'].min()} ~ {self.df_valid['預測日期'].max()}")
        
        # 定義模型列表（排除非預測列）
        exclude_cols = ['預測日期', '星期', '觀測數據', '开单日期']
        self.model_columns = [col for col in self.df_merged.columns if col not in exclude_cols]
        
        print(f"   - 評估模型數量: {len(self.model_columns)} 個")
        
        return self
    
    def calculate_metrics(self):
        """計算各模型的誤差指標"""
        print("\n" + "=" * 100)
        print("📈 模型誤差指標計算")
        print("=" * 100)
        
        results = []
        for model in self.model_columns:
            # 只使用有觀測數據的行
            y_true = self.df_valid['觀測數據'].values
            y_pred = self.df_valid[model].values
            
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
        
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values('MAE(件)')
        
        print("\n🏆 模型效能排名（按MAE由低到高）:")
        print(self.results_df[['模型名稱', 'MAE(件)', 'RMSE(件)', 'MAPE(%)', 'R²', '偏誤(Bias)']].head(10).to_string(index=False))
        
        return self.results_df
    
    def categorize_models(self):
        """對模型進行分類並計算各類別統計"""
        print("\n" + "=" * 100)
        print("📊 各類型模型效能對比")
        print("=" * 100)
        
        model_categories = {
            '移動平均': ['SMA-3', 'SMA-7', 'SMA-15', 'SMA-30'],
            '加權移動平均': ['WMA-7', 'WMA-15'],
            '指數平滑': ['ES-α=0.3', 'ES-α=0.5', 'ES-α=0.7'],
            '雙指數平滑': ['Holt-α=0.5,β=0.3', 'Holt-α=0.6,β=0.2'],
            '季節性預測': ['Seasonal-周期7天', 'Seasonal-周期15天', 'Seasonal-周期30天']
        }
        
        category_results = []
        for category, models in model_categories.items():
            category_models = self.results_df[self.results_df['模型名稱'].isin(models)]
            if len(category_models) > 0:
                category_results.append({
                    '模型類型': category,
                    '平均MAE': round(category_models['MAE(件)'].mean(), 0),
                    '平均MAPE(%)': round(category_models['MAPE(%)'].mean(), 1),
                    '最佳模型': category_models.iloc[category_models['MAE(件)'].argmin()]['模型名稱'],
                    '最佳MAE': category_models['MAE(件)'].min()
                })
        
        self.category_df = pd.DataFrame(category_results)
        self.category_df = self.category_df.sort_values('平均MAE')
        print(self.category_df.to_string(index=False))
        
        return self.category_df
    
    def get_best_model_info(self):
        """獲取最佳模型信息"""
        if self.results_df is None or len(self.results_df) == 0:
            return None, None, None, None
        
        best_row = self.results_df.iloc[0]
        best_model = best_row['模型名稱']
        best_mae = best_row['MAE(件)']
        best_mape = best_row['MAPE(%)']
        best_r2 = best_row['R²']
        
        print("\n" + "=" * 100)
        print(f"🏆 最佳預測模型: {best_model}")
        print("=" * 100)
        print(f"   ✅ MAE (平均絕對誤差): {best_mae:.0f} 件")
        print(f"   ✅ RMSE (均方根誤差): {best_row['RMSE(件)']:.0f} 件")
        print(f"   ✅ MAPE (平均絕對百分比誤差): {best_mape:.1f}%")
        print(f"   ✅ R² (決定係數): {best_r2:.3f}")
        print(f"   📊 偏誤: {best_row['偏誤(Bias)']:.0f} 件")
        
        return best_model, best_mae, best_mape, best_r2
    
    def plot_performance_charts(self, best_model, best_mae, best_mape, best_r2):
        """生成預測 vs 實際對比圖表"""
        print("\n" + "=" * 100)
        print("📈 生成預測 vs 實際對比圖表")
        print("=" * 100)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'車縫組 - 預測模型效能評估 (最佳模型: {best_model})', fontsize=14, fontweight='bold')
        
        # 圖1: 最佳模型預測 vs 實際值
        ax = axes[0, 0]
        df_sorted = self.df_valid.sort_values('預測日期')
        ax.plot(df_sorted['預測日期'], df_sorted['觀測數據'], 'b-o', linewidth=2, markersize=6, label='實際觀測值')
        ax.plot(df_sorted['預測日期'], df_sorted[best_model], 'r--s', linewidth=2, markersize=6, label=f'{best_model} 預測值')
        ax.set_title(f'最佳模型: {best_model}\nMAE={best_mae:.0f}, MAPE={best_mape:.1f}%, R²={best_r2:.3f}', 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('交貨數 (件)')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 圖2: 誤差柱狀圖（所有模型）
        ax = axes[0, 1]
        top_models = self.results_df.head(8)
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
        
        # 圖3: 預測誤差分布（最佳模型）
        ax = axes[1, 0]
        errors = self.df_valid['觀測數據'].values - self.df_valid[best_model].values
        ax.hist(errors, bins=15, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零誤差線')
        ax.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
                   label=f'平均誤差: {np.mean(errors):.0f}')
        ax.set_title(f'{best_model} 預測誤差分布', fontsize=11, fontweight='bold')
        ax.set_xlabel('預測誤差 (件)')
        ax.set_ylabel('頻次')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 圖4: 散點圖 - 預測 vs 實際
        ax = axes[1, 1]
        ax.scatter(self.df_valid['觀測數據'], self.df_valid[best_model], alpha=0.7, s=80, 
                   c='steelblue', edgecolors='navy')
        max_val = max(self.df_valid['觀測數據'].max(), self.df_valid[best_model].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美預測線')
        ax.set_title(f'{best_model} - 預測值 vs 實際值', fontsize=11, fontweight='bold')
        ax.set_xlabel('實際交貨數 (件)')
        ax.set_ylabel('預測交貨數 (件)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '模型效能評估對比圖.png'), dpi=150, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ 效能評估圖表已儲存")
        
        return fig
    
    def plot_all_models_comparison(self):
        """生成所有模型詳細預測對比圖"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('車縫組 - 所有模型預測 vs 實際對比', fontsize=14, fontweight='bold')
        
        n_models = min(len(self.model_columns), 6)
        for idx in range(n_models):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            model = self.model_columns[idx]
            
            df_sorted = self.df_valid.sort_values('預測日期')
            ax.plot(df_sorted['預測日期'], df_sorted['觀測數據'], 'b-o', linewidth=2, markersize=5, label='實際觀測值')
            ax.plot(df_sorted['預測日期'], df_sorted[model], 'r--s', linewidth=2, markersize=5, label=f'{model}')
            
            model_mae = self.results_df[self.results_df['模型名稱'] == model]['MAE(件)'].values[0]
            ax.set_title(f'{model} (MAE: {model_mae:.0f}件)', fontsize=10, fontweight='bold')
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
    
    def print_recommendations(self, best_model, best_mae, best_mape):
        """打印模型推薦建議"""
        print("\n" + "=" * 100)
        print("💡 模型推薦建議")
        print("=" * 100)
        
        # 找出第二、第三好的模型
        second_best = self.results_df.iloc[1]['模型名稱'] if len(self.results_df) > 1 else None
        third_best = self.results_df.iloc[2]['模型名稱'] if len(self.results_df) > 2 else None
        
        print(f"""
📌 最佳模型: {best_model}
   - 優點: 預測誤差最小，準確度最高
   - 適用場景: 一般生產規劃與預測
   - MAE: {best_mae:.0f}件 | MAPE: {best_mape:.1f}%

📌 次佳模型: {second_best}
   - 可作為備用方案或模型集成使用

📌 各類型模型推薦:
""")
        
        if self.category_df is not None:
            for _, row in self.category_df.iterrows():
                print(f"   • {row['模型類型']}: {row['最佳模型']} (MAE: {row['最佳MAE']:.0f}件)")
        
        # 找出SMA模型
        sma_models = self.results_df[self.results_df['模型名稱'].str.contains('SMA')]
        best_sma = sma_models.iloc[0]['模型名稱'] if len(sma_models) > 0 else 'SMA模型'
        
        # 找出季節性模型
        seasonal_models = self.category_df[self.category_df['模型類型'] == '季節性預測'] if self.category_df is not None else pd.DataFrame()
        best_seasonal = seasonal_models['最佳模型'].values[0] if len(seasonal_models) > 0 else '季節性模型'
        
        print(f"""
📊 模型選擇建議:
   1. 優先使用 {best_model} 進行預測
   2. 若需要預測穩定性，可考慮 {best_sma}
   3. 若數據有季節性趨勢，可參考 {best_seasonal}
   4. 建議使用多模型集成（取平均）來降低單一模型風險
""")
    
    def save_excel_report(self):
        """保存Excel詳細報告"""
        excel_file = os.path.join(self.output_folder, '模型效能評估報告.xlsx')
        
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 模型排名表
                self.results_df.to_excel(writer, sheet_name='模型排名', index=False)
                
                # 各模型類型匯總
                if self.category_df is not None:
                    self.category_df.to_excel(writer, sheet_name='模型類型匯總', index=False)
                
                # 預測vs實際明細
                detail_df = self.df_valid[['預測日期', '星期', '觀測數據'] + self.model_columns].copy()
                detail_df.to_excel(writer, sheet_name='預測vs實際明細', index=False)
                
                # 每日誤差分析
                error_analysis = self.df_valid[['預測日期', '星期', '觀測數據']].copy()
                for model in self.model_columns:
                    error_analysis[f'{model}_誤差'] = self.df_valid['觀測數據'] - self.df_valid[model]
                    error_analysis[f'{model}_絕對誤差'] = np.abs(self.df_valid['觀測數據'] - self.df_valid[model])
                error_analysis.to_excel(writer, sheet_name='每日誤差分析', index=False)
                
                # 模型推薦
                second_best = self.results_df.iloc[1]['模型名稱'] if len(self.results_df) > 1 else 'N/A'
                sma_models = self.results_df[self.results_df['模型名稱'].str.contains('SMA')]
                best_sma = sma_models.iloc[0]['模型名稱'] if len(sma_models) > 0 else 'SMA-7'
                best_sma_mae = sma_models.iloc[0]['MAE(件)'] if len(sma_models) > 0 else 'N/A'
                
                seasonal_models = self.category_df[self.category_df['模型類型'] == '季節性預測'] if self.category_df is not None else pd.DataFrame()
                best_seasonal = seasonal_models['最佳模型'].values[0] if len(seasonal_models) > 0 else 'N/A'
                best_seasonal_mae = seasonal_models['最佳MAE'].values[0] if len(seasonal_models) > 0 else 'N/A'
                
                recommendation = pd.DataFrame({
                    '推薦類型': ['最佳準確度', '備用方案', '最穩定(移動平均)', '季節性考量', '保守預測', '樂觀預測'],
                    '推薦模型': [
                        self.results_df.iloc[0]['模型名稱'],
                        second_best,
                        best_sma,
                        best_seasonal,
                        self.results_df[self.results_df['平均預測'] == self.results_df['平均預測'].min()]['模型名稱'].values[0] if len(self.results_df) > 0 else 'N/A',
                        self.results_df[self.results_df['平均預測'] == self.results_df['平均預測'].max()]['模型名稱'].values[0] if len(self.results_df) > 0 else 'N/A'
                    ],
                    'MAE(件)': [
                        self.results_df.iloc[0]['MAE(件)'],
                        self.results_df.iloc[1]['MAE(件)'] if len(self.results_df) > 1 else 'N/A',
                        best_sma_mae,
                        best_seasonal_mae,
                        self.results_df[self.results_df['平均預測'] == self.results_df['平均預測'].min()]['MAE(件)'].values[0] if len(self.results_df) > 0 else 'N/A',
                        self.results_df[self.results_df['平均預測'] == self.results_df['平均預測'].max()]['MAE(件)'].values[0] if len(self.results_df) > 0 else 'N/A'
                    ]
                })
                recommendation.to_excel(writer, sheet_name='模型推薦', index=False)
            
            print(f"\n✅ Excel詳細報告已儲存: {excel_file}")
        except Exception as e:
            print(f"\n⚠️ Excel儲存失敗: {e}")
    
    def print_final_summary(self, best_model, best_mae, best_mape, best_r2):
        """打印最終評估摘要"""
        second_best = self.results_df.iloc[1]['模型名稱'] if len(self.results_df) > 1 else 'N/A'
        third_best = self.results_df.iloc[2]['模型名稱'] if len(self.results_df) > 2 else 'N/A'
        
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
    
    def run(self):
        """執行完整的效能評估流程"""
        # 1. 加載數據
        self.load_data()
        
        # 2. 篩選與合併
        self.filter_and_merge()
        
        # 3. 清理數據
        self.clean_data()
        
        # 4. 計算評估指標
        self.calculate_metrics()
        
        # 5. 模型分類
        self.categorize_models()
        
        # 6. 獲取最佳模型信息
        best_model, best_mae, best_mape, best_r2 = self.get_best_model_info()
        
        # 7. 生成對比圖表
        self.plot_performance_charts(best_model, best_mae, best_mape, best_r2)
        
        # 8. 生成所有模型對比圖
        self.plot_all_models_comparison()
        
        # 9. 打印推薦建議
        self.print_recommendations(best_model, best_mae, best_mape)
        
        # 10. 保存Excel報告
        self.save_excel_report()
        
        # 11. 打印最終摘要
        self.print_final_summary(best_model, best_mae, best_mape, best_r2)
        
        print("\n✨ 模型效能評估完成！")
        
        return self.results_df


# ============================================
# 使用範例
# ============================================
if __name__ == "__main__":
    # 配置參數
    forecast_excel_path = r'C:\Users\Dell\Desktop\git_demo\车缝组_多模型预测对比报告(排除周日)\00_多模型预测综合报告.xlsx'
    production_excel_path = r'C:\Users\Dell\Downloads\生产每日交货清单 - 複製.xlsx'
    
    # 款號列表（P12857相關）
    items = ['P1174-7', 'P11774-8', 'P11775-7', 'P11775-8', 'P11776-7',
             'P11776-8', 'P12092-4', 'P12092-5', 'P12093-5', 'P12093-4',
             'P12094-4', 'P12094-5', 'P12297-5', 'P12297-4']
    
    # 創建評估器實例並運行
    evaluator = ModelEvaluator(
        forecast_excel_path=forecast_excel_path,
        forecast_sheet_name='所有模型預測匯總',
        production_excel_path=production_excel_path,
        production_sheet_name='生产每日交货清单',
        items=items,
        output_folder='模型效能評估報告'
    )
    
    # 執行完整評估
    results = evaluator.run()
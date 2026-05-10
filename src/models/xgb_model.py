"""
XGBoost 预测模型 + 滚动训练 + 信号生成
利用已有15+因子捕捉非线性关系
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.factors import FactorRegistry
from src.factors.builtin import register_builtin_factors

register_builtin_factors()
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """预测结果"""
    prob_up: float         # 上涨概率
    prob_down: float       # 下跌概率
    signal: str            # long / short / hold
    confidence: float      # 置信度 0-1
    feature_importance: dict  # 特征重要性


class XGBPredictor:
    """
    XGBoost 预测器
    
    使用已有因子作为特征，预测未来价格方向。
    滚动训练避免过拟合。
    """
    
    def __init__(
        self,
        # 模型参数
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        reg_lambda: float = 1.0,        # L2正则化（防过拟合）
        reg_alpha: float = 0.5,         # L1正则化
        min_child_weight: int = 3,
        # 预测参数
        forward_period: int = 24,        # 预测未来N根K线
        threshold_long: float = 0.55,    # 做多阈值
        threshold_short: float = 0.45,   # 做空阈值
        # 训练参数
        retrain_interval: int = 168,     # 每N根K线重新训练（1周）
        min_train_size: int = 720,       # 最少训练数据（30天）
        val_ratio: float = 0.2,          # 验证集比例
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_child_weight = min_child_weight
        
        self.forward_period = forward_period
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        
        self.retrain_interval = retrain_interval
        self.min_train_size = min_train_size
        self.val_ratio = val_ratio
        
        # 模型和标准化器
        self.model = None
        self.scaler = StandardScaler()
        self._feature_names = None
        self._last_train_index = -1  # 上次训练的索引
        
        # 因子实例
        self.factors = {}
        for name in FactorRegistry.list_factors():
            f = FactorRegistry.get(name)
            if f and f.lookback <= 50:  # 排除需要太长时间窗口的
                self.factors[name] = f
    
    def _build_features(self, data: pd.DataFrame, idx: int) -> Optional[np.ndarray]:
        """构建特征向量"""
        features = []
        
        for name, factor in self.factors.items():
            try:
                values = factor.calculate(data)
                if idx >= len(values):
                    features.extend([0.0, 0.0, 0.0])
                    continue
                
                raw = values.iloc[idx]
                if pd.isna(raw) or np.isinf(raw):
                    features.extend([0.0, 0.0, 0.0])
                    continue
                
                window = values.iloc[max(0, idx-100):idx+1].dropna()
                if len(window) < 10:
                    features.extend([0.0, 0.0, 0.0])
                    continue
                
                mu = window.mean()
                sd = window.std()
                norm = (raw - mu) / sd if sd > 0 else 0.0
                features.append(float(np.clip(norm, -5, 5)))
                
                raw_val = float(raw if np.isfinite(raw) else 0.0)
                features.append(np.clip(raw_val, -1e8, 1e8))
                
                if idx >= 5:
                    prev = values.iloc[idx-5]
                    if pd.notna(prev) and abs(prev) > 1e-8:
                        chg = (raw - prev) / abs(prev)
                        features.append(float(np.clip(chg, -5, 5)))
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
                    
            except:
                features.extend([0.0, 0.0, 0.0])
        
        # 价格特征
        try:
            price = data['close'].iloc[idx]
            if idx >= 20:
                ma20 = data['close'].iloc[idx-20:idx+1].mean()
                features.append(float((price - ma20) / ma20))
                mom5 = data['close'].iloc[idx-5] if idx >= 5 else price
                features.append(float((price - mom5) / mom5))
                mom20 = data['close'].iloc[idx-20] if idx >= 20 else price
                features.append(float((price - mom20) / mom20))
            else:
                features.extend([0.0, 0.0, 0.0])
        except:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def _build_labels(self, data: pd.DataFrame, idx: int) -> Optional[int]:
        """构建标签（涨跌方向，只使用未来数据）"""
        if idx + self.forward_period >= len(data):
            return None
        
        future_price = data['close'].iloc[idx + self.forward_period]
        current_price = data['close'].iloc[idx]
        
        if future_price > current_price * 1.005:  # 涨>0.5%
            return 1
        elif future_price < current_price * 0.995:  # 跌>0.5%
            return 0
        else:
            return None  # 横盘，忽略
    
    def _create_training_data(self, data: pd.DataFrame, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
        """创建训练数据"""
        X_list, y_list = [], []
        
        for idx in range(start, end):
            feat = self._build_features(data, idx)
            label = self._build_labels(data, idx)
            
            if feat is not None and label is not None:
                X_list.append(feat.flatten())
                y_list.append(label)
        
        if not X_list:
            return np.array([]), np.array([])
        
        return np.array(X_list), np.array(y_list)
    
    def _train(self, X, y):
        """训练模型"""
        if len(X) < 100:
            return False
        
        # 划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_ratio, random_state=42
        )
        
        # 计算类别权重（平衡涨跌样本）
        n_up = (y_train == 1).sum()
        n_down = (y_train == 0).sum()
        scale_pos_weight = n_down / max(n_up, 1)
        
        # 创建模型
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=20,
            random_state=42,
        )
        
        # 训练
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        # 记录特征名称（只在第一次）
        if self._feature_names is None:
            self._feature_names = []
            for name, factor in self.factors.items():
                self._feature_names.extend([f"{name}_norm", f"{name}_raw", f"{name}_chg"])
            self._feature_names.extend(['price_ma_dev', 'momentum_5', 'momentum_20'])
        
        # 验证集准确率
        val_acc = (self.model.predict(X_val) == y_val).mean()
        logger.info(f"Model trained: {len(X_train)} samples, val_acc={val_acc:.2%}")
        
        return True
    
    def predict(self, data: pd.DataFrame, idx: int) -> Optional[PredictionResult]:
        """预测"""
        # 检查是否需要重新训练
        if self.model is None or idx - self._last_train_index > self.retrain_interval:
            train_start = max(0, idx - self.min_train_size * 3)
            X, y = self._create_training_data(data, train_start, idx)
            
            if len(X) >= self.min_train_size:
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                if self._train(X_scaled, y):
                    self._last_train_index = idx
        
        if self.model is None:
            return None
        
        # 构建特征
        feat = self._build_features(data, idx)
        if feat is None:
            return None
        
        feat_scaled = self.scaler.transform(feat)
        
        # 预测概率
        proba = self.model.predict_proba(feat_scaled)[0]
        prob_up = proba[1] if len(proba) > 1 else 0.5
        prob_down = proba[0] if len(proba) > 0 else 0.5
        
        # 生成信号
        confidence = abs(prob_up - 0.5) * 2  # 0-1
        if prob_up > self.threshold_long:
            signal = 'long'
        elif prob_up < self.threshold_short:
            signal = 'short'
        else:
            signal = 'hold'
        
        # 特征重要性
        importance = {}
        if self._feature_names and len(self._feature_names) == self.model.feature_importances_.shape[0]:
            for name, imp in zip(self._feature_names, self.model.feature_importances_):
                importance[name] = imp
        
        return PredictionResult(
            prob_up=prob_up,
            prob_down=prob_down,
            signal=signal,
            confidence=confidence,
            feature_importance=importance,
        )


class XGBStrategy:
    """基于XGBoost预测的交易策略"""
    
    def __init__(self, **kwargs):
        self.predictor = XGBPredictor(**kwargs)
        self.stop_loss = 0.03
        self.take_profit = 0.06
    
    def generate_signal(self, data: pd.DataFrame, idx: int,
                       position_side: str = 'flat',
                       entry_price: float = None, **kw) -> str:
        """生成交易信号"""
        if idx < 360:  # 最少15天数据
            return 'hold'
        
        # 预测
        result = self.predictor.predict(data, idx)
        if result is None:
            return 'hold'
        
        current_price = data['close'].iloc[idx]
        
        # 进入信号
        if position_side == 'flat':
            if result.signal == 'long' and result.confidence > 0.5:
                return 'long'
            if result.signal == 'short' and result.confidence > 0.5:
                return 'short'
        
        # 退出信号
        elif position_side == 'long':
            if entry_price:
                if current_price < entry_price * (1 - self.stop_loss):
                    return 'close'
                if current_price > entry_price * (1 + self.take_profit):
                    return 'close'
            if result.signal == 'short' and result.confidence > 0.5:
                return 'close'
        
        elif position_side == 'short':
            if entry_price:
                if current_price > entry_price * (1 + self.stop_loss):
                    return 'close'
                if current_price < entry_price * (1 - self.take_profit):
                    return 'close'
            if result.signal == 'long' and result.confidence > 0.5:
                return 'close'
        
        return 'hold'
    
    def get_importance(self) -> Dict[str, float]:
        """获取特征重要性（用于因子分析）"""
        if self.predictor.model is None:
            return {}
        
        importance = {}
        if self.predictor._feature_names:
            for name, imp in zip(self.predictor._feature_names,
                                 self.predictor.model.feature_importances_):
                importance[name] = imp
        
        # 按重要性排序
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])

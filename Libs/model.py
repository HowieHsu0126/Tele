import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (HalvingGridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.svm import SVC
import torch
import torch.nn.functional as F

class Models:
    @staticmethod
    def handle_imbalance(X, y, logger):
        logger.info("Handling class imbalance using SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info("Class imbalance handled successfully.")
        return X_resampled, y_resampled

    @staticmethod
    def feature_selection(X, y, logger, n=20):
        logger.info("Starting feature selection...")
        embed = RandomForestClassifier(n_estimators=100, random_state=42)
        embed.fit(X, y)

        # 结合各个方法的特征重要性
        logger.info("Combining feature importances from all methods...")
        feature_scores = embed.feature_importances_

        # 选择最重要的特征
        logger.info("Selecting top features...")
        indices = np.argsort(feature_scores)[-n:]  # 选择排名前n的特征
        selected_features = X.columns[indices]

        # 使用选中的特征重新构建数据集
        X_selected = X[selected_features]
        logger.info("Feature selection completed successfully.")
        return X_selected, selected_features

    @staticmethod
    def evaluate_models(X, y, logger, k=3, tune_hyperparameters=True):
        logger.info("Evaluating multiple models...")
        models = {
            'RandomForest': RandomForestClassifier(),
            # 'ExtraTrees': ExtraTreesClassifier(),
            # 'AdaBoost': AdaBoostClassifier(),
            # 'HistGradientBoosting': HistGradientBoostingClassifier(),
            # 'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            # 'LightGBM': lgb.LGBMClassifier(),
            # 'CatBoost': CatBoostClassifier(verbose=0),
        }

        search_spaces = {
            'RandomForest': {
                'n_estimators': [100],
                'criterion': ["gini", "entropy"],
                'max_features': np.arange(0.05, 1.01, 0.05),
                'min_samples_split': range(2, 21),
                'min_samples_leaf':  range(1, 21),
                'bootstrap': [True, False],
            },
            # 'ExtraTrees': {
            #     'n_estimators': [100],
            #     'criterion': ["gini", "entropy"],
            #     'max_features': np.arange(0.05, 1.01, 0.05),
            #     'min_samples_split': range(2, 21),
            #     'min_samples_leaf': range(1, 21),
            #     'bootstrap': [True, False],
            # },
            # 'AdaBoost': {
            #     'n_estimators': [50, 100, 200],
            #     'learning_rate': [0.01, 0.1, 1, 10]
            # },
            # 'HistGradientBoosting': {
            #     'learning_rate': [0.01, 0.1, 0.2],
            #     'max_iter': [100, 200, 300]
            # },
            # 'XGBoost': {
            #     'n_estimators': [100],
            #     'max_depth': range(1, 11),
            #     'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            #     'subsample': np.arange(0.05, 1.01, 0.05),
            #     'min_child_weight': range(1, 21),
            #     'verbosity': [0],
            # },
            # 'LightGBM': {
            #     'n_estimators': [100, 200, 300],
            #     'learning_rate': [0.01, 0.1, 0.2],
            #     'num_leaves': [31, 62, 127]
            # },
            # 'CatBoost': {
            #     'iterations': [100, 200, 300],
            #     'learning_rate': [0.01, 0.1, 0.2],
            #     'depth': [3, 6, 9]
            # },
        }

        best_models = []
        for name, model in models.items():
            if tune_hyperparameters and name in search_spaces:
                logger.info(f"Evaluating {name} with hyperparameter search...")
                search = HalvingGridSearchCV(
                    model, search_spaces[name], scoring='f1', cv=5, factor=2, random_state=42, n_jobs=-1)
                search.fit(X, y)
                best_model = search.best_estimator_
                mean_score = search.best_score_
            else:
                logger.info(f"Evaluating {name} without hyperparameter search...")
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
                best_model = model.fit(X, y)
                mean_score = np.mean(cv_scores)

            best_models.append((name, best_model, mean_score))
            logger.info(f"{name} - Best F1 Score: {mean_score}")

        # 排序模型得分并选择最佳的k个模型
        sorted_models = sorted(best_models, key=lambda x: x[2], reverse=True)
        selected_models = [(name, model) for name, model, score in sorted_models[:k]]
        logger.info("Best models selected for ensemble.")
        return selected_models

    @staticmethod
    def adversarial_training(model, X, y, epsilon=0.01):
        """
        对抗训练函数，使用FGSM生成对抗样本并加入训练集
        """
        model.fit(X, y)  # 首先用原始数据训练模型

        X_adv = X.copy()  # 创建对抗样本的副本
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        X_tensor.requires_grad = True
        outputs = model.predict_proba(X)[:, 1]
        loss = F.cross_entropy(torch.tensor(outputs).reshape(-1, 1), y_tensor)

        model.zero_grad()
        loss.backward()
        X_adv += epsilon * X_tensor.grad.sign().numpy()

        # 将原始样本和对抗样本结合起来训练模型
        X_combined = np.vstack((X, X_adv))
        y_combined = np.hstack((y, y))

        model.fit(X_combined, y_combined)
        return model

    @staticmethod
    def train_model(X, y, best_models, logger):
        logger.info("Training ensemble model with adversarial training...")
        estimators = [(name, Models.adversarial_training(model, X, y)) for name, model in best_models]
        meta_learner = LogisticRegression(max_iter=1000)
        ensemble_model = StackingClassifier(
            estimators=estimators, final_estimator=meta_learner, cv=5)
        cv_scores = cross_val_score(ensemble_model, X, y, cv=5, scoring='f1')
        logger.info(f'Ensemble Model - Cross-Validated F1 Score: {cv_scores.mean()}')
        ensemble_model.fit(X, y)
        logger.info("Ensemble model training completed successfully.")
        return ensemble_model

    @staticmethod
    def pseudo_labeling(model, X_train, y_train, X_val, logger, threshold=0.8):
        logger.info("Applying pseudo-labeling...")
        pseudo_labels = model.predict_proba(X_val)[:, 1]
        high_confidence_indices = np.where(pseudo_labels >= threshold)[0]
        X_pseudo = X_val.iloc[high_confidence_indices]
        y_pseudo = np.ones(X_pseudo.shape[0])
        logger.info(f"Pseudo-labeling completed with {len(y_pseudo)} pseudo-labels.")

        X_combined = pd.concat([X_train, X_pseudo])
        y_combined = np.concatenate([y_train, y_pseudo])

        return X_combined, y_combined

    @staticmethod
    def predict_and_save(model, X_val, validation_res, selected_features, output_path, logger):
        logger.info("Predicting and saving results...")

        X_val_selected = X_val[selected_features]
        val_pred = model.predict(X_val_selected)

        # 对每个 msisdn 进行聚合，如果任何一次预测为1，则最终预测为1
        validation_res['is_sa'] = val_pred
        final_predictions = validation_res.groupby('msisdn')['is_sa'].max().reset_index()

        final_predictions.to_csv(output_path, index=False)
        logger.info("Results saved successfully.")
    
    def run_pipeline(self, X_train, y_train, X_val, validation_res, output_path, logger, tune_hyperparameters=True, adversarial_training=True):
        # 处理类不平衡
        X_resampled, y_resampled = self.handle_imbalance(X_train, y_train, logger)
        
        # 特征选择
        X_selected, selected_features = self.feature_selection(X_resampled, y_resampled, logger)
        
        # 模型评估并选择最佳模型
        best_models = self.evaluate_models(X_selected, y_resampled, logger, tune_hyperparameters=tune_hyperparameters)
        
        # 模型训练
        if adversarial_training:
            final_model = self.train_model(X_selected, y_resampled, best_models, logger)
        else:
            estimators = [(name, model) for name, model in best_models]
            meta_learner = LogisticRegression(max_iter=1000)
            final_model = StackingClassifier(estimators=estimators, final_estimator=meta_learner, cv=5)
            final_model.fit(X_selected, y_resampled)
        
        # 伪标签
        X_combined, y_combined = self.pseudo_labeling(final_model, X_selected, y_resampled, X_val, logger)
        
        # 再次训练模型（包括伪标签）
        if adversarial_training:
            final_model = self.train_model(X_combined, y_combined, best_models, logger)
        else:
            final_model.fit(X_combined, y_combined)
        
        # 预测并保存结果
        self.predict_and_save(final_model, X_val, validation_res, selected_features, output_path, logger)


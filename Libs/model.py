import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier, StackingClassifier)
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


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
        logger.info("Combining feature impotences from all methods...")
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
    def evaluate_models(X, y, logger, k=3):
        logger.info("Evaluating multiple models...")
        models = {
            'RandomForest': RandomForestClassifier(),
            'ExtraTrees': ExtraTreesClassifier(),
            # 'GradientBoosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'HistGradientBoosting': HistGradientBoostingClassifier(),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(),
            'CatBoost': CatBoostClassifier(verbose=0)
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
            'ExtraTrees': {
                'n_estimators': [100],
                'criterion': ["gini", "entropy"],
                'max_features': np.arange(0.05, 1.01, 0.05),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21),
                'bootstrap': [True, False],
            },
            # 'GradientBoosting': {
            #     'n_estimators': [100],
            #     'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            #     'max_depth': range(1, 11),
            #     'min_samples_split': range(2, 21),
            #     'min_samples_leaf': range(1, 21),
            #     'subsample': np.arange(0.05, 1.01, 0.05),
            #     'max_features': np.arange(0.05, 1.01, 0.05),
            # },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1, 10]
            },
            'HistGradientBoosting': {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_iter': [100, 200, 300]
            },
            'XGBoost': {
                'n_estimators': [100],
                'max_depth': range(1, 11),
                'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                'subsample': np.arange(0.05, 1.01, 0.05),
                'min_child_weight': range(1, 21),
                'verbosity': [0],
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 62, 127]
            },
            'CatBoost': {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 6, 9]
            }
        }

        best_models = []
        for name, model in models.items():
            logger.info(f"Evaluating {name} with hyperparameter search...")
            search = HalvingGridSearchCV(
                model, search_spaces[name], scoring='f1', cv=5, factor=2, random_state=42, n_jobs=-1)
            search.fit(X, y)
            best_model = search.best_estimator_
            mean_score = search.best_score_
            best_models.append((name, best_model, mean_score))
            logger.info(f"{name} - Best F1 Score: {mean_score}")

        # 排序模型得分并选择最佳的k个模型
        sorted_models = sorted(best_models, key=lambda x: x[2], reverse=True)
        selected_models = [(name, model)
                           for name, model, score in sorted_models[:k]]
        logger.info("Best models selected for ensemble.")
        return selected_models

    @staticmethod
    def train_model(X, y, best_models, logger):
        logger.info("Training ensemble model...")
        estimators = [(name, model) for name, model in best_models]
        meta_learner = LogisticRegression(max_iter=1000)
        ensemble_model = StackingClassifier(
            estimators=estimators, final_estimator=meta_learner, cv=5)
        cv_scores = cross_val_score(ensemble_model, X, y, cv=5, scoring='f1')
        logger.info(
            f'Ensemble Model - Cross-Validated F1 Score: {cv_scores.mean()}')
        ensemble_model.fit(X, y)
        logger.info("Ensemble model training completed successfully.")
        return ensemble_model

    @staticmethod
    def predict_and_save(model, X_val, validation_res, selected_features, output_path, logger):
        logger.info("Predicting and saving results...")
        
        X_val_selected = X_val[selected_features]
        val_pred = model.predict(X_val_selected)

        # 对每个 msisdn 进行聚合，如果任何一次预测为1，则最终预测为1
        validation_res['is_sa'] = val_pred
        final_predictions = validation_res.groupby(
            'msisdn')['is_sa'].max().reset_index()

        final_predictions.to_csv(output_path, index=False)
        logger.info("Results saved successfully.")

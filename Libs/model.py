import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.feature_selection import (RFE, SelectKBest, VarianceThreshold,
                                       chi2, f_classif, mutual_info_classif)
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
    def feature_selection(X, y, logger):
        logger.info("Starting feature selection...")
        embed_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        embed_rf.fit(X, y)
        rf_importances = embed_rf.feature_importances_

        embed_gbdt = GradientBoostingClassifier(
            n_estimators=100, random_state=42)
        embed_gbdt.fit(X, y)
        gbdt_importances = embed_gbdt.feature_importances_

        embed_etr = ExtraTreesClassifier(n_estimators=100, random_state=42)
        embed_etr.fit(X, y)
        etr_importances = embed_etr.feature_importances_

        embed_logreg = LogisticRegression(max_iter=1000)
        embed_logreg.fit(X, y)
        logreg_importances = np.abs(embed_logreg.coef_[0])

        # 结合各个方法的特征重要性
        logger.info("Combining feature importances from all methods...")
        feature_scores = (rf_importances + gbdt_importances +
                          etr_importances + logreg_importances) / 4.0

        # 选择最重要的特征
        logger.info("Selecting top features...")
        indices = np.argsort(feature_scores)[-20:]  # 选择排名前20的特征
        selected_features = X.columns[indices]

        # 使用选中的特征重新构建数据集
        X_selected = X[selected_features]
        logger.info("Feature selection completed successfully.")
        return X_selected, selected_features

    @staticmethod
    def evaluate_models(X, y, logger):
        logger.info("Evaluating multiple models...")
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
        }
        model_scores = {}
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
            mean_score = cv_scores.mean()
            model_scores[name] = (model, mean_score)
            logger.info(f"{name} - Cross-Validated F1 Score: {mean_score}")

        # 排序模型得分并选择最佳的k个模型
        sorted_models = sorted(model_scores.items(),
                               key=lambda item: item[1][1], reverse=True)
        best_k_models = [(name, model[0])
                         for name, model in sorted_models[:3]]  # 选择最好的三个模型
        logger.info("Best models selected for ensemble.")
        return best_k_models

    @staticmethod
    def train_model(X, y, best_models, logger):
        logger.info("Training ensemble model using VotingClassifier...")
        estimators = [(name, model) for name, model in best_models]
        voting_model = VotingClassifier(estimators=estimators, voting='soft')
        cv_scores = cross_val_score(voting_model, X, y, cv=5, scoring='f1')
        logger.info(
            f'Ensemble Model - Cross-Validated F1 Score: {cv_scores.mean()}')
        voting_model.fit(X, y)
        logger.info("Ensemble model training completed successfully.")
        return voting_model

    @staticmethod
    def predict_and_save(model, X_val, validation_res, selected_features, output_path, logger):
        logger.info("Predicting and saving results...")
        X_val_selected = X_val[selected_features]
        val_pred = model.predict(X_val_selected)

        # 对每个 msisdn 进行聚合，以确保每个 msisdn 只有一个预测结果
        validation_res['is_sa'] = val_pred
        final_predictions = validation_res.groupby(
            'msisdn')['is_sa'].agg(lambda x: x.mode()[0]).reset_index()

        final_predictions.to_csv(output_path, index=False)
        logger.info("Results saved successfully.")

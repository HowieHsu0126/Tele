from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

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
        logger.info("Performing feature selection using RandomForestClassifier...")
        select_model = RandomForestClassifier(n_estimators=100, random_state=42)
        select_model.fit(X, y)
        selector = SelectFromModel(select_model, prefit=True)
        X_selected = selector.transform(X)
        logger.info("Feature selection completed successfully.")
        return X_selected, selector

    @staticmethod
    def evaluate_models(X, y, logger, k=3):
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
        sorted_models = sorted(model_scores.items(), key=lambda item: item[1][1], reverse=True)
        best_k_models = [model[0] for model in sorted_models[:k]]  # 选择最好的k个模型
        logger.info("Best models selected for ensemble.")
        return best_k_models

    @staticmethod
    def train_model(X, y, best_models, logger):
        logger.info("Training ensemble model using VotingClassifier...")
        estimators = [(name, model) for name, model in best_models]
        ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
        cv_scores = cross_val_score(ensemble_model, X, y, cv=5, scoring='f1')
        logger.info(f'Ensemble Model - Cross-Validated F1 Score: {cv_scores.mean()}')
        ensemble_model.fit(X, y)
        logger.info("Ensemble model training completed successfully.")
        return ensemble_model

    @staticmethod
    def predict_and_save(model, X_val, validation_res, selector, output_path, logger):
        logger.info("Predicting and saving results...")
        X_val_selected = selector.transform(X_val)
        val_pred = model.predict(X_val_selected)
        # 对每个 msisdn 进行聚合，以确保每个 msisdn 只有一个预测结果
        validation_res['is_sa'] = val_pred
        final_predictions = validation_res.groupby('msisdn')['is_sa'].agg(lambda x: x.mode()[0]).reset_index()
        
        final_predictions.to_csv(output_path, index=False)
        logger.info("Results saved successfully.")

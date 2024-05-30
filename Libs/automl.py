from tpot import TPOTClassifier


class AutoML:
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

    @staticmethod
    def run_automl(X, y, logger):
        logger.info("Running AutoML...")
        pipeline_optimizer = TPOTClassifier(generations=5, population_size=50,
                                            verbosity=2, random_state=42, scoring='f1',)

        pipeline_optimizer.fit(X, y)
        pipeline_optimizer.export('auto_pipeline.py')
        logger.info(f"F1 Score: {pipeline_optimizer.score(X, y)}")
        logger.info("AutoML training completed successfully.")
        logger.info(f"Best pipeline: {pipeline_optimizer.fitted_pipeline_}")
        return pipeline_optimizer.fitted_pipeline_

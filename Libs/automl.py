from tpot import TPOTClassifier


class AutoML:
    @staticmethod
    def run_automl(X, y, logger):
        logger.info("Running AutoML...")
        pipeline_optimizer = TPOTClassifier(generations=5, population_size=50,
                                            verbosity=2, random_state=42, scoring='f1')

        pipeline_optimizer.fit(X, y)
        pipeline_optimizer.export('auto_pipeline.py')
        logger.info(f"F1 Score: {pipeline_optimizer.score(X, y)}")
        logger.info("AutoML training completed successfully.")
        logger.info(f"Best pipeline: {pipeline_optimizer.fitted_pipeline_}")
        return pipeline_optimizer.fitted_pipeline_

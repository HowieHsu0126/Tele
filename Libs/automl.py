from tpot import TPOTClassifier


class AutoML:
    @staticmethod
    def run_automl(X, y, logger):
        logger.info("Running AutoML...")
        tpot = TPOTClassifier(generations=5, population_size=50,
                              verbosity=2, random_state=42, scoring='f1')
        tpot.fit(X, y)
        logger.info("AutoML training completed successfully.")
        logger.info(f"Best pipeline: {tpot.fitted_pipeline_}")
        return tpot.fitted_pipeline_

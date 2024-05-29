import pandas as pd
from sklearn.model_selection import train_test_split
from utils import Utils

import pandas as pd

class Datasets:
    @staticmethod
    def load_data(logger):
        logger.info("Loading data...")
        train_res = pd.read_csv('/sda/xuhaowei/Research/Tele/Input/raw/train.csv', low_memory=False)
        train_ans = pd.read_csv('/sda/xuhaowei/Research/Tele/Input/raw/labels.csv')
        validation_res = pd.read_csv('/sda/xuhaowei/Research/Tele/Input/raw/val.csv', low_memory=False)
        logger.info("Data loaded successfully.")
        return train_res, train_ans, validation_res

    @staticmethod
    def merge_data(train_res, train_ans, logger):
        logger.info("Merging training data and labels...")
        train_data = pd.merge(train_res, train_ans, on='msisdn')
        logger.info("Data merged successfully.")
        return train_data

    @staticmethod
    def preprocess_data(train_data, validation_res, logger):
        logger.info("Preprocessing data...")
        Utils.convert_to_datetime(train_data, ['start_time', 'end_time', 'open_datetime'])
        Utils.convert_to_datetime(validation_res, ['start_time', 'end_time', 'open_datetime'])
        train_data['update_time'] = pd.to_datetime(train_data['update_time'], errors='coerce')
        train_data['date'] = pd.to_datetime(train_data['date'], errors='coerce')
        train_data['date_c'] = pd.to_datetime(train_data['date_c'], format='%Y%m%d', errors='coerce')
        Utils.create_time_features(train_data)
        Utils.create_time_features(validation_res)
        fields_to_convert = ['visit_area_code', 'called_code', 'phone1_loc_city', 'phone1_loc_province', 'phone2_loc_city', 'phone2_loc_province']
        for field in fields_to_convert:
            train_data[field] = train_data[field].astype(str)
            validation_res[field] = validation_res[field].astype(str)
        logger.info("Data preprocessed successfully.")
        return train_data, validation_res

    @staticmethod
    def feature_engineering(train_data, validation_res, categorical_features, logger):
        logger.info("Performing feature engineering...")
        train_data = pd.get_dummies(train_data, columns=categorical_features)
        validation_res = pd.get_dummies(validation_res, columns=categorical_features)
        validation_res = validation_res.reindex(columns=train_data.columns, fill_value=0)
        logger.info("Feature engineering completed successfully.")
        return train_data, validation_res

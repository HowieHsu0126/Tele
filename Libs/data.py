import pandas as pd
from sklearn.model_selection import train_test_split
from utils import Utils

import pandas as pd


class Datasets:
    @staticmethod
    def load_data(logger):
        logger.info("Loading data...")
        train_res = pd.read_csv(
            '/sda/xuhaowei/Research/Tele/Input/raw/train.csv', low_memory=False)
        train_ans = pd.read_csv(
            '/sda/xuhaowei/Research/Tele/Input/raw/labels.csv')
        validation_res = pd.read_csv(
            '/sda/xuhaowei/Research/Tele/Input/raw/val.csv', low_memory=False)
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
        Utils.convert_to_datetime(
            train_data, ['start_time', 'end_time', 'open_datetime'])
        Utils.convert_to_datetime(
            validation_res, ['start_time', 'end_time', 'open_datetime'])
        train_data['update_time'] = pd.to_datetime(
            train_data['update_time'], errors='coerce')
        train_data['date'] = pd.to_datetime(
            train_data['date'], errors='coerce')
        train_data['date_c'] = pd.to_datetime(
            train_data['date_c'], format='%Y%m%d', errors='coerce')
        Utils.create_time_features(train_data)
        Utils.create_time_features(validation_res)
        fields_to_convert = ['visit_area_code', 'called_code', 'phone1_loc_city',
                             'phone1_loc_province', 'phone2_loc_city', 'phone2_loc_province']
        for field in fields_to_convert:
            train_data[field] = train_data[field].astype(str)
            validation_res[field] = validation_res[field].astype(str)
        logger.info("Data preprocessed successfully.")
        return train_data, validation_res

    @staticmethod
    def feature_engineering(train_data, validation_res, categorical_features, logger):
        logger.info("Performing feature engineering...")
        # 时间特征提取
        logger.info("Extracting time features...")
        train_data['start_hour'] = train_data['start_time'].dt.hour
        train_data['start_dayofweek'] = train_data['start_time'].dt.dayofweek
        train_data['is_weekend'] = train_data['start_dayofweek'].apply(
            lambda x: 1 if x >= 5 else 0)
        train_data['is_working_hour'] = train_data['start_hour'].apply(
            lambda x: 1 if 9 <= x <= 18 else 0)

        validation_res['start_hour'] = validation_res['start_time'].dt.hour
        validation_res['start_dayofweek'] = validation_res['start_time'].dt.dayofweek
        validation_res['is_weekend'] = validation_res['start_dayofweek'].apply(
            lambda x: 1 if x >= 5 else 0)
        validation_res['is_working_hour'] = validation_res['start_hour'].apply(
            lambda x: 1 if 9 <= x <= 18 else 0)

        train_data = pd.get_dummies(train_data, columns=categorical_features)
        validation_res = pd.get_dummies(
            validation_res, columns=categorical_features)

        # 构建新特征
        logger.info("Constructing new features...")
        train_data['call_fee_rate'] = train_data['cfee'] / \
            (train_data['call_duration'] + 1)
        validation_res['call_fee_rate'] = validation_res['cfee'] / \
            (validation_res['call_duration'] + 1)

        train_data['long_call_rate'] = train_data['lfee'] / \
            (train_data['call_duration'] + 1)
        validation_res['long_call_rate'] = validation_res['lfee'] / \
            (validation_res['call_duration'] + 1)

        train_data['total_fee'] = train_data['cfee'] + train_data['lfee']
        validation_res['total_fee'] = validation_res['cfee'] + \
            validation_res['lfee']

        # 确保训练集和验证集具有相同的列
        logger.info(
            "Aligning validation data columns with training data columns...")
        validation_res = validation_res.reindex(
            columns=train_data.columns, fill_value=0)

        logger.info("Feature engineering completed successfully.")

        return train_data, validation_res

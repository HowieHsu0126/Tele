import datetime
import logging

import pandas as pd


class Logger:
    @staticmethod
    def setup_logger(project_name='baseline'):
        logger = logging.getLogger(project_name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(
            f'/sda/xuhaowei/Research/Tele/Output/logs/{project_name}_{datetime.datetime.now()}.log')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger


class Utils:
    @staticmethod
    def convert_to_datetime(df, cols, fmt='%Y%m%d%H%M%S'):
        for col in cols:
            df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')

    @staticmethod
    def create_time_features(df):
        df['call_duration_minutes'] = df['call_duration'] / 60
        df['start_hour'] = df['start_time'].dt.hour
        df['start_dayofweek'] = df['start_time'].dt.dayofweek

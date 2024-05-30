import datetime
import logging

import numpy as np
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

    @staticmethod
    def clean_data(df):
        # 通用的数据清理，将包含非数字字符的字段替换为合理的默认值
        cols_to_clean = [
            'home_area_code', 'visit_area_code', 'called_home_code', 'called_code'
        ]
        for col in cols_to_clean:
            df[col] = df[col].apply(
                lambda x: '0000' if not str(x).isdigit() else x)
        return df

    @staticmethod
    def reduce_mem_usage(df, logger):
        numerics = ['int16', 'int32', 'int64',
                    'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
                end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df

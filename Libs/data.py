import category_encoders as ce
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.decomposition import PCA
from utils import Utils


class Datasets:

    @staticmethod
    def load_and_clean_data(file_paths, logger):
        logger.info("Loading and cleaning data...")
        data_frames = {}
        for name, path in file_paths.items():
            df = pd.read_csv(path, low_memory=False)
            df = Utils.reduce_mem_usage(df, logger)
            data_frames[name] = df
        logger.info("Data loaded and cleaned successfully.")
        return data_frames

    @staticmethod
    def merge_data(train_res, train_ans, key='msisdn'):
        return pd.merge(train_res, train_ans, on=key)

    @staticmethod
    def preprocess_data(data_frames, date_fields, str_fields, logger):
        logger.info("Preprocessing data...")

        for name, df in data_frames.items():
            Utils.convert_to_datetime(df, date_fields)
            df[str_fields] = df[str_fields].astype(str)
            df = Utils.create_time_features(df)

        logger.info("Data preprocessed successfully.")
        return data_frames

    @staticmethod
    def feature_engineering(data_frames, logger):
        logger.info("Performing feature engineering...")

        for name, df in data_frames.items():
            df = Datasets.extract_time_features(df)
            df = Datasets.construct_new_features(df)

        train_data = data_frames['train_data']
        validation_res = data_frames['validation_res']

        train_data, validation_res = Datasets.encode_features(
            train_data, validation_res)

        data_frames['train_data'] = train_data
        data_frames['validation_res'] = validation_res

        for name, df in data_frames.items():
            df = Datasets.normalize_features(df)
            data_frames[name] = df

        logger.info("Feature engineering completed successfully.")
        return data_frames

    @staticmethod
    def extract_time_features(df):
        df['start_hour'] = df['start_time'].dt.hour
        df['start_dayofweek'] = df['start_time'].dt.dayofweek
        df['is_weekend'] = df['start_dayofweek'].apply(
            lambda x: 1 if x >= 5 else 0)
        df['is_working_hour'] = df['start_hour'].apply(
            lambda x: 1 if 9 <= x <= 18 else 0)
        df['call_duration'] = (
            df['end_time'] - df['start_time']).dt.total_seconds()
        df['call_start_hour'] = df['start_time'].dt.hour
        df['call_end_hour'] = df['end_time'].dt.hour
        return df

    @staticmethod
    def construct_new_features(df):
        # 通话费用率和长途费用率
        df['call_fee_rate'] = df['cfee'] / (df['call_duration'] + 1)
        df['long_call_rate'] = df['lfee'] / (df['call_duration'] + 1)
        df['total_fee'] = df['cfee'] + df['lfee']

        # 疑似类型标志
        suspect_types = {3, 5, 6, 9, 11, 12, 17}
        df['is_suspect'] = df['phone1_type'].apply(
            lambda x: 1 if x in suspect_types else 0)

        # 通话次数和唯一通话者数
        df['call_count'] = df.groupby('msisdn')['msisdn'].transform('count')
        df['unique_callers'] = df.groupby(
            'msisdn')['other_party'].transform('nunique')

        # 各类通话费用总和
        df['total_cfee'] = df.groupby('msisdn')['cfee'].transform('sum')
        df['total_lfee'] = df.groupby('msisdn')['lfee'].transform('sum')

        # 平均通话时长
        df['avg_call_duration'] = df.groupby(
            'msisdn')['call_duration'].transform('mean')

        # 将类别变量进行数值编码再计算比例特征
        df['call_event_encoded'] = df['call_event'].apply(
            lambda x: 1 if x == 'call_src' else 0)
        df['call_event_ratio'] = df.groupby(
            'msisdn')['call_event_encoded'].transform('mean')

        df['video_call_ratio'] = df.groupby(
            'msisdn')['ismultimedia'].transform('mean')

        df['roam_type_encoded'] = df['roam_type'].astype('category').cat.codes
        df['roam_type_ratio'] = df.groupby(
            'msisdn')['roam_type_encoded'].transform('mean')

        df['a_serv_type_encoded'] = df['a_serv_type'].apply(
            lambda x: 1 if x == '01' else 0)
        df['caller_ratio'] = df.groupby(
            'msisdn')['a_serv_type_encoded'].transform('mean')

        # 地域特征
        df['is_same_home_area'] = (
            df['home_area_code'] == df['called_home_code']).astype(int)
        df['is_same_visit_area'] = (
            df['visit_area_code'] == df['called_code']).astype(int)
        df['home_area_call_count'] = df.groupby(
            'home_area_code')['msisdn'].transform('count')
        df['visit_area_call_count'] = df.groupby('visit_area_code')[
            'msisdn'].transform('count')
        df['called_home_area_call_count'] = df.groupby(
            'called_home_code')['msisdn'].transform('count')
        df['called_visit_area_call_count'] = df.groupby(
            'called_code')['msisdn'].transform('count')

        return df

    @staticmethod
    def encode_features(train_data, validation_res):
        # Label Encoding
        label_features = ['call_event', 'a_serv_type']
        for feature in label_features:
            le = LabelEncoder()
            train_data[feature] = le.fit_transform(train_data[feature])
            validation_res[feature] = le.transform(validation_res[feature])

        # Ordinal Encoding
        ordinal_features = ['roam_type']
        for feature in ordinal_features:
            oe = OrdinalEncoder()
            train_data[feature] = oe.fit_transform(train_data[[feature]])
            validation_res[feature] = oe.transform(validation_res[[feature]])

        # OneHot Encoding
        onehot_features = ['ismultimedia', 'home_area_code', 'visit_area_code',
                           'called_home_code', 'called_code', 'long_type1', 'phone1_type', 'phone2_type']
        onehot_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown='ignore')
        onehot_encoded_train = onehot_encoder.fit_transform(
            train_data[onehot_features])
        onehot_encoded_val = onehot_encoder.transform(
            validation_res[onehot_features])
        onehot_encoded_columns = onehot_encoder.get_feature_names_out(
            onehot_features)

        onehot_encoded_train_df = pd.DataFrame(
            onehot_encoded_train, columns=onehot_encoded_columns)
        onehot_encoded_val_df = pd.DataFrame(
            onehot_encoded_val, columns=onehot_encoded_columns)

        train_data = pd.concat([train_data.reset_index(
            drop=True), onehot_encoded_train_df.reset_index(drop=True)], axis=1)
        validation_res = pd.concat([validation_res.reset_index(
            drop=True), onehot_encoded_val_df.reset_index(drop=True)], axis=1)

        train_data.drop(columns=onehot_features, inplace=True)
        validation_res.drop(columns=onehot_features, inplace=True)

        # Frequency Encoding
        freq_features = ['a_product_id']
        for feature in freq_features:
            freq_encoding = train_data[feature].value_counts().to_dict()
            train_data[feature +
                       '_freq'] = train_data[feature].map(freq_encoding)
            validation_res[feature +
                           '_freq'] = validation_res[feature].map(freq_encoding)

        # Target Encoding
        target_features = ['phone1_loc_city', 'phone2_loc_city',
                           'phone1_loc_province', 'phone2_loc_province']
        target_encoder = ce.TargetEncoder(cols=target_features)
        train_data[target_features] = target_encoder.fit_transform(
            train_data[target_features], train_data['is_sa'])
        validation_res[target_features] = target_encoder.transform(
            validation_res[target_features])

        return train_data, validation_res

    @staticmethod
    def normalize_features(df):
        numerical_features = ['call_duration', 'cfee', 'lfee', 'start_hour', 'start_dayofweek', 'call_duration_minutes',
                              'call_fee_rate', 'long_call_rate', 'total_fee', 'call_count', 'unique_callers',
                              'total_cfee', 'total_lfee', 'avg_call_duration', 'call_event_ratio',
                              'video_call_ratio', 'roam_type_ratio', 'caller_ratio', 'home_area_call_count',
                              'visit_area_call_count', 'called_home_area_call_count', 'called_visit_area_call_count']
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        return df

    @staticmethod
    def prepare_datasets(train_data, validation_res):
        X = train_data.drop(columns=['msisdn', 'is_sa', 'start_time', 'end_time', 'open_datetime',
                                     'update_time', 'date', 'date_c'])
        y = train_data['is_sa']
        X_val = validation_res.drop(columns=['msisdn', 'start_time', 'end_time', 'open_datetime',
                                             'update_time', 'date', 'date_c'])
        return X, y, X_val

    @staticmethod
    def handle_imbalance(X, y, logger):
        logger.info("Handling class imbalance using SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info("Class imbalance handled successfully.")
        return X_resampled, y_resampled

    @staticmethod
    def feature_selection(X, y, logger, n=500):
        logger.info("Starting feature selection...")
        embed = RandomForestClassifier(n_estimators=100, random_state=42)
        embed.fit(X, y)

        # 结合各个方法的特征重要性
        logger.info("Combining feature importances from all methods...")
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
    def apply_pca(X, logger, n_components=100):
        logger.info("Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        logger.info("PCA applied successfully.")
        return X_pca

    @staticmethod
    def adversarial_validation(X, X_val, logger):
        logger.info("Starting adversarial validation...")
        X['is_train'] = 1
        X_val['is_train'] = 0
        combined_data = pd.concat([X, X_val], axis=0)
        y = combined_data['is_train']

        X_train, X_test, y_train, y_test = train_test_split(
            combined_data, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        logger.info(f'Adversarial validation AUC: {auc}')

        X.drop(columns=['is_train'], inplace=True)
        X_val.drop(columns=['is_train'], inplace=True)

        if auc > 0.7:
            logger.warning(
                "Adversarial validation AUC is high. Training and validation sets have different distributions.")
        else:
            logger.info(
                "Adversarial validation AUC is low. Training and validation sets have similar distributions.")

    def run_pipeline(self, file_paths, date_fields, str_fields, logger):
        data_frames = self.load_and_clean_data(file_paths, logger)
        train_res, train_ans, validation_res = data_frames[
            'train_res'], data_frames['train_ans'], data_frames['validation_res']
        train_data = self.merge_data(train_res, train_ans)
        data_frames = self.preprocess_data(
            {'train_data': train_data, 'validation_res': validation_res}, date_fields, str_fields, logger)
        data_frames = self.feature_engineering(data_frames, logger)
        train_data, validation_res = data_frames['train_data'], data_frames['validation_res']
        X, y, X_val = self.prepare_datasets(train_data, validation_res)
        # self.adversarial_validation(X, X_val, logger)
        # 处理类不平衡
        X_resampled, y_resampled = self.handle_imbalance(X, y, logger)

        # 特征选择
        X_resampled_selected, selected_features = self.feature_selection(
            X_resampled, y_resampled, logger, n=1000)
        X_val_selected = X_val[selected_features]

        # Apply PCA after feature selection
        X_resampled_pca = self.apply_pca(
            X_resampled_selected, n_components=500)
        X_val_pca = self.apply_pca(X_val_selected, n_components=500)

        return X_resampled_pca, y_resampled, X_val_pca, validation_res

from data import Datasets
from model import Models
from utils import Logger
import datetime

def main():
    logger = Logger.setup_logger()

    # 数据加载
    train_res, train_ans, validation_res = Datasets.load_data(logger)

    # 数据合并
    train_data = Datasets.merge_data(train_res, train_ans, logger)

    # 数据预处理
    train_data, validation_res = Datasets.preprocess_data(train_data, validation_res, logger)

    # 特征工程
    categorical_features = ['call_event', 'ismultimedia', 'home_area_code', 'visit_area_code', 'called_home_code', 
                            'called_code', 'a_serv_type', 'long_type1', 'roam_type', 'a_product_id', 'phone1_type', 
                            'phone2_type', 'phone1_loc_city', 'phone1_loc_province', 'phone2_loc_city', 'phone2_loc_province']
    train_data, validation_res = Datasets.feature_engineering(train_data, validation_res, categorical_features, logger)

    # 准备特征和标签
    X = train_data.drop(columns=['msisdn', 'is_sa', 'start_time', 'end_time', 'open_datetime', 'update_time', 'date', 'date_c'])
    y = train_data['is_sa']

    # 处理样本不平衡
    X_resampled, y_resampled = Models.handle_imbalance(X, y, logger)

    # 特征选择
    X_resampled_selected, selector = Models.feature_selection(X_resampled, y_resampled, logger)

    # 模型训练和评估
    best_models = Models.evaluate_models(X_resampled_selected, y_resampled, logger)
    model = Models.train_model(X_resampled_selected, y_resampled, best_models, logger)

    # 预测并保存结果
    X_val = validation_res.drop(columns=['msisdn', 'is_sa', 'start_time', 'end_time', 'open_datetime', 'update_time', 'date', 'date_c'])
    Models.predict_and_save(model, X_val, validation_res, selector, f'/sda/xuhaowei/Research/Tele/Output/submissions/prediction_{datetime.datetime.now()}.csv', logger)

if __name__ == "__main__":
    main()


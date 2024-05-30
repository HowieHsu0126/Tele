import datetime

from automl import AutoML
from data import Datasets
from model import Models
from utils import Logger


def main(project_name='baseline'):
    logger = Logger.setup_logger(project_name)

    # 数据文件路径
    file_paths = {
        'train_res': '/sda/xuhaowei/Research/Tele/Input/raw/train.csv',
        'train_ans': '/sda/xuhaowei/Research/Tele/Input/raw/labels.csv',
        'validation_res': '/sda/xuhaowei/Research/Tele/Input/raw/val.csv'
    }

    # 日期字段和字符串字段列表
    date_fields = ['start_time', 'end_time',
                   'open_datetime', 'update_time', 'date', 'date_c']
    str_fields = ['visit_area_code', 'called_code', 'called_home_code',
                  'phone1_loc_city', 'phone1_loc_province',
                  'phone2_loc_city', 'phone2_loc_province']

    # 运行数据处理流水线
    dataset = Datasets()
    X, y, X_val, validation_res = dataset.run_pipeline(
        file_paths, date_fields, str_fields, logger)

    # 处理样本不平衡
    X_resampled, y_resampled = Models.handle_imbalance(X, y, logger)

    # 特征选择
    X_resampled_selected, selector = Models.feature_selection(
        X_resampled, y_resampled, logger)

    # 模型训练和评估
    best_models = Models.evaluate_models(
        X_resampled_selected, y_resampled, logger)
    model = Models.train_model(
        X_resampled_selected, y_resampled, best_models, logger)
    # model = AutoML.run_automl(X_resampled_selected, y_resampled, logger)

    # 预测并保存结果
    Models.predict_and_save(model, X_val, validation_res, selector,
                            f'/sda/xuhaowei/Research/Tele/Output/submissions/prediction_{datetime.datetime.now()}.csv', logger)
    # AutoML.predict_and_save(model, X_val, validation_res, selector,
    #                         f'/sda/xuhaowei/Research/Tele/Output/submissions/prediction_{datetime.datetime.now()}.csv', logger)


if __name__ == "__main__":
    main('baseline')

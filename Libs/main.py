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
        'validation_res': '/sda/xuhaowei/Research/Tele/Input/raw/val.csv',
    }

    # 日期字段和字符串字段列表
    date_fields = ['start_time', 'end_time',
                   'open_datetime', 'update_time', 'date', 'date_c']
    str_fields = ['visit_area_code', 'called_code', 'called_home_code',
                  'phone1_loc_city', 'phone1_loc_province',
                  'phone2_loc_city', 'phone2_loc_province']

    output_path = f'/sda/xuhaowei/Research/Tele/Output/submissions/prediction_{datetime.datetime.now()}.csv'
    
    # 运行数据处理流水线
    data_controller = Datasets()
    X, y, X_val, validation_res = data_controller.run_pipeline(
        file_paths, date_fields, str_fields, logger)

    # model_controller = Models()
    # model_controller.run_pipeline(
    #     X, y, X_val, validation_res, output_path, logger, tune_hyperparameters=False, adversarial_training=False)

    model = AutoML.run_automl(X, y, logger)
    AutoML.predict_and_save(model, X_val, validation_res, output_path, logger)


if __name__ == "__main__":
    main('automl')

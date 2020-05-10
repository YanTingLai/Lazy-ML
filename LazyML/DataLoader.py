import os
import pandas as pd


class GetData:
    def __init__(self, data_name):
        """
        讀取預設資料集
        :param data_name: 資料集名稱
        """
        self.dt_name = os.listdir('../data')
        self.data_types = ['train', 'test']
        self.wd = os.getcwd()
        self.target = data_name

    def load_data(self):
        if self.target == 'fraud':
            type_dict = {
                'train': {'user_id': 'str', 'device_id': 'str', 'age': 'int', 'sex': 'int', 'browser': 'str', 'source': 'str', 'country': 'str', 'purchase_value': 'int', 'signup_time': 'str', 'purchase_time': 'str', 'TXCNT': 'int', 'class': 'int'},
                'test': {'user_id': 'str', 'device_id': 'str', 'age': 'int', 'sex': 'int', 'browser': 'str', 'source': 'str', 'country': 'str', 'purchase_value': 'int', 'signup_time': 'str', 'purchase_time': 'str', 'TXCNT': 'int'}
            }
            parse_dates = ['signup_time', 'purchase_time']

            file_path = {x: os.path.join('..', 'data', self.target, x) + '.zip' for x in self.data_types}
            data_set = {x: pd.read_csv(file_path[x],
                                       header=0,
                                       keep_default_na=True,
                                       dtype=type_dict[x],
                                       encoding='utf-8',
                                       parse_dates=parse_dates)
                        for x in self.data_types}
            return data_set

        elif self.target == 'mnist':
            file_path = {x: os.path.join('..', 'data', self.target, 'mnist-' + x) + '.zip' for x in self.data_types}
            data_set = {x: pd.read_csv(file_path[x],
                                       header=0,
                                       keep_default_na=True,
                                       encoding='utf-8',
                                       )
                        for x in self.data_types}
            return data_set
        else:
            raise(
                'Dataset does not exists.'
            )


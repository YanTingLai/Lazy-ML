# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 11:26:54 2020

@author: 10356
"""
import os
from collections import Counter
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, make_scorer, classification_report, fbeta_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from catboost import CatBoostClassifier, Pool, cv
from numpy.random import RandomState
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import hyperopt
from hyperopt import tpe, hp
import xgboost as xgb
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


def grid_search(model, parameters, x_, y_):
    print()
    print("Grid Searching...")
    acc_scorer = make_scorer(fbeta_score, beta=1)
    grid_obj = GridSearchCV(model, parameters, scoring=acc_scorer, n_jobs=-1, cv=5)
    grid_obj = grid_obj.fit(x_, y_.values.ravel())
    return grid_obj.best_estimator_


def classifier(model, grid_search_flag, smote_flag, x_train, y_train, x_valid, y_valid, x_test, cols):
    clf_name = model.__class__.__name__
    parameters = parameter_set(clf_name)
    print("=" * 30)
    if smote_flag:
        clf_name += '-Smote'

    if grid_search_flag:
        clf_name += '-Grid'
        print(clf_name)
        print()
        print("Grid Search Parameters:")
        print(parameters)
        model_ = grid_search(model, parameters, x_train, y_train)
    else:
        model_ = model
        print(clf_name)

    print()
    print("Model Parameters:")
    for k, v in model_.get_params().items():
        print(k, v)
    print()

    model_.fit(x_train, y_train.values.ravel())
    predict_ = model_.predict(x_valid)
    if clf_name == 'XGBClassifier' or clf_name == 'XGBClassifierGrid':
        predict_ = [value for value in predict_]

    importances_ = model_.feature_importances_[:10]
    indices_ = np.argsort(importances_)[::-1]
    print("Feature ranking:")

    for f_ in range(len(importances_)):
        print("%3d. %25s  (%f)" % (f_ + 1, cols[indices_[f_]], importances_[indices_[f_]]))

    f_score_ = fbeta_score(y_valid, predict_, beta=1, average='binary')
    plot_result(y_valid, predict_)

    # 將預測結果存入ensemble list備用
    global ensemble_valid, ensemble_test
    ensemble_valid[clf_name] = predict_
    submit_output = model_.predict(x_test)
    ensemble_test[clf_name] = [value for value in submit_output] if 'XGBClassifier' in clf_name else submit_output

    return [clf_name, f_score_*100]


def parameter_set(clf_name):
    """
    記錄各演算法超參數
    :param clf_name: 演算法名稱
    :return: 演算法超參數
    """
    if clf_name == 'RandomForestClassifier':
        parameters = {
            'n_estimators': [50, 100, 150, 500],
            'criterion': ['entropy', 'gini'],
            'max_depth': list(range(3, 10)),
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
             }
    if clf_name == 'DecisionTreeClassifier':
        parameters = {
            'criterion': ['entropy', 'gini'],
            'splitter': ['best', 'random'],
            'max_depth': list(range(3, 10)),
            # 'class_weight': [{0: 1, 1: 1}, {0: 1, 1: 2}]
             }
    if clf_name == 'GradientBoostingClassifier':
        parameters = {
            "loss": ["deviance", 'exponential'],
            "learning_rate": [0.1, 0.2, 0.5, 0.7],
            # "min_samples_split": list(range(2, 5)),
            # "min_samples_leaf": list(range(1, 5)),
            'max_depth': list(range(3, 10)),
            "criterion": ["friedman_mse",  "mae", 'mse'],
            "n_estimators": [10, 100, 150],
             }
    if clf_name == 'XGBClassifier':
        parameters = {
            # General parameters
            'booster': ['gbtree'],
            # Parameters of Tree booster
            'eta': [0.1, 0.15, 0.2],
            'gamma': [0.1, 0.15, 0.2, 0.5, 0.7],
            'max_depth': list(range(3, 10)),
            'learning_rate': [0.1, 0.15, 0.2, 0.5, 0.7],
            'num_parallel_tree': [1, 2, 3],
            'objective': ['binary:logistic'],
            'eval_metric': ['auc', 'aucpr', 'map']
        }
    return parameters


def get_data(data_cate: str):
    """
    讀取資料檔
    :param data_cate: 資料集類別，train, test
    :return: Pandas DataFrame
    """
    type_dict = {
        'train': {'user_id': 'str', 'device_id': 'str', 'age': 'int', 'sex': 'int', 'browser': 'str', 'source': 'str', 'country': 'str', 'purchase_value': 'int', 'signup_time': 'str', 'purchase_time': 'str', 'TXCNT': 'int', 'class': 'int'},
        'test': {'user_id': 'str', 'device_id': 'str', 'age': 'int', 'sex': 'int', 'browser': 'str', 'source': 'str', 'country': 'str', 'purchase_value': 'int', 'signup_time': 'str', 'purchase_time': 'str', 'TXCNT': 'int'}
    }
    parse_dates = ['signup_time', 'purchase_time']
    file_path = os.path.join(os.getcwd(), data_cate) + '.csv'
    data_set = pd.read_csv(file_path,
                           header=0,
                           keep_default_na=True,
                           dtype=type_dict[data_cate],
                           encoding='utf-8',
                           parse_dates=parse_dates)
    return data_set


def clean_data(train, test):
    """
    資料處理流程
    :param train: 測試集
    :param test: 驗證集
    :return: x_train, x_valid, y_train, y_valid, x_test, cols, user_id
    """
    y_train = train['class'].copy()
    train.drop(['class'], 1, inplace=True)

    user_id = test['user_id'].copy()

    # 將資料集合併做資料處理
    train['training'] = 1
    test['training'] = 0
    all_df = pd.concat([train, test])

    # 處理城市變數
    all_df.loc[pd.isna(all_df.country), 'country'] = 'unknown'
    # 小城市合併成其他
    country_list = [k for k, v in Counter(train_df.country.tolist()).items() if v < 5]
    all_df.loc[all_df.country.isin(country_list), 'country'] = 'others'

    # 處理device_id
    # 發現device_id與user_id是一對多關係
    # 且device_id重複使用的詐騙率很高
    # 所以標註device_id是否重複使用
    device_id = all_df.device_id.tolist()
    dup = []
    for k, v in Counter(device_id).items():
        if v > 1:
            dup += [k]
    device_id_dup = [id_ in dup for id_ in device_id]
    all_df['dup_device'] = device_id_dup

    # 產生衍生性變數:(purchase_time - signup_time) 註冊日與交易日天數差異
    all_df['diff_days'] = (all_df.purchase_time - all_df.signup_time).dt.seconds

    # 新增月份、小時、每周第幾天與交易天
    all_df['signup_month'] = all_df.signup_time.dt.month
    all_df['signup_hour'] = all_df.signup_time.dt.hour
    all_df['signup_dayofweek'] = all_df.signup_time.dt.dayofweek
    all_df['purchase_day'] = all_df.purchase_time.dt.day
    all_df['purchase_month'] = all_df.purchase_time.dt.month
    all_df['purchase_hour'] = all_df.purchase_time.dt.hour
    all_df['dayofweek'] = all_df.purchase_time.dt.dayofweek

    # 產生device_id單天消費次數
    # 有發現詐騙標註常發生在單日內
    # 因此算出device_id單日有多少交易量
    # 但此變數加入模型後表現變非常差
    # 還需再觀察

    # tt = all_df[['device_id', 'purchase_time', 'purchase_month', 'purchase_day']].copy()
    # aa = tt.groupby(['device_id', 'purchase_month', 'purchase_day'], as_index=False).count()
    # aa.columns = ['device_id', 'purchase_month', 'purchase_day', 'dup_purchase']
    # aa['dup_purchase'] = [0 if x_ == 1 else 1 for x_ in aa['dup_purchase'].tolist()]
    # # print(tt[tt['device_id'] == 'b00884bec1cbe'])
    # # print(aa[aa['device_id'] == 'b00884bec1cbe'])
    # all_df = pd.merge(left=all_df,
    #                   right=aa,
    #                   left_on=['device_id', 'purchase_month', 'purchase_day'],
    #                   right_on=['device_id', 'purchase_month', 'purchase_day']
    #                   )
    # all_df['dup_purchase_2'] = all_df['dup_purchase'] * all_df['dup_device']

    print(all_df.columns)
    # 刪掉多餘變數
    all_df.drop(['device_id', 'signup_time', 'purchase_time', 'user_id'], 1, inplace=True)

    # 類別變數轉dummy
    # 日期衍伸變數不轉dummy，ensemble後F1更高。
    dummy_list = [
        'browser',
        'source',
        'country',
        # 'signup_month',
        # 'signup_hour',
        # 'signup_dayofweek',
        # 'purchase_month',
        # 'purchase_hour',
        # 'dayofweek',
        # 'dup_purchase'
    ]
    all_df = pd.get_dummies(all_df, prefix=dummy_list, columns=dummy_list)

    # 再將資料切為訓練、測試
    x_train = all_df[all_df.training == 1].copy()
    x_test = all_df[all_df.training == 0].copy()
    x_train.drop(['training'], 1, inplace=True)
    x_test.drop(['training'], 1, inplace=True)

    cols = x_train.columns
    # 標準化
    # Feature Scaling
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=0.20,
        random_state=5566
    )
    return x_train, x_valid, y_train, y_valid, x_test, cols, user_id


def plot_result(y_t, y_p):
    """
    畫混淆矩陣與分類結果
    :param y_t: 實際y
    :param y_p: 預測y
    :return: None
    """
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_t, y_p))
    print()
    print("Classification Report:")
    print(classification_report(y_t, y_p))
    pass


def catboost_classifier(x_train, y_train, x_valid, y_valid, x_test, cols, hyper_tune, smote):
    hyper_algo = tpe.suggest
    d_train = Pool(x_train, y_train)
    d_val = Pool(x_valid, y_valid)

    def get_catboost_params(space_):
        params = dict()
        params['learning_rate'] = space_['learning_rate']
        params['depth'] = int(space_['depth'])
        params['l2_leaf_reg'] = space_['l2_leaf_reg']
        params['rsm'] = space_['rsm']
        return params

    def hyperopt_objective(space_):
        params = get_catboost_params(space_)
        sorted_params = sorted(space.items(), key=lambda z: z[0])
        params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
        print('Params: {}'.format(params_str))

        model_ = CatBoostClassifier(iterations=100,
                                    learning_rate=params['learning_rate'],
                                    depth=int(params['depth']),
                                    loss_function='Logloss',
                                    use_best_model=True,
                                    eval_metric='AUC',
                                    l2_leaf_reg=params['l2_leaf_reg'],
                                    random_seed=5566,
                                    verbose=False
                                    )
        cv_ = cv(d_train, model_.get_params())
        best_accuracy = np.max(cv_['test-AUC-mean'])
        return 1 - best_accuracy

    if hyper_tune:
        space = {
            'depth': hp.quniform("depth", 4, 7, 1),
            'rsm': hp.uniform('rsm', 0.75, 1.0),
            'learning_rate': hp.loguniform('learning_rate', -3.0, -0.7),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
        }
        trials = hyperopt.Trials()
        best = hyperopt.fmin(
            hyperopt_objective,
            space=space,
            algo=hyper_algo,
            max_evals=50,
            trials=trials,
            rstate=RandomState(5566)
        )
        print('-' * 50)
        print('The best params:')
        print(best)
        print('\n\n')

        model = CatBoostClassifier(
            l2_leaf_reg=int(best['l2_leaf_reg']),
            learning_rate=best['learning_rate'],
            depth=best['depth'],
            iterations=50,
            eval_metric='AUC',
            random_seed=42,
            loss_function='Logloss',
            verbose=False
        )
    else:
        model = CatBoostClassifier(
            l2_leaf_reg=6,
            learning_rate=0.24,
            depth=8,
            iterations=100,
            eval_metric='AUC',
            random_seed=42,
            loss_function='Logloss',
            verbose=False
        )

    cv_data = cv(pool=d_train,
                 params=model.get_params(),
                 nfold=5,
                 verbose=False
                 )

    model.fit(x_train, y_train)

    print('Best validation AUC score: {:.2f}±{:.2f} on step {}'.format(
        np.max(cv_data['test-AUC-mean']),
        cv_data['test-AUC-std'][np.argmax(cv_data['test-AUC-mean'])],
        np.argmax(cv_data['test-AUC-mean'])
    ))

    predict_ = model.predict(x_valid)

    f_score_ = fbeta_score(y_valid, predict_, beta=1, average='binary')
    smote_flag = '-Smote' if smote else ''
    global log
    log = log.append(pd.DataFrame([['Catboost' + smote_flag, f_score_ * 100]], columns=log_cols))

    plot_result(y_valid, predict_)
    global ensemble_test, ensemble_valid
    ensemble_valid['Catboost'] = predict_
    ensemble_test['Catboost'] = model.predict(x_test)
    feature_importances = model.get_feature_importance(Pool(x_train, y_train))
    feature_names = cols
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
        if score > 0.1:
            print("%25s  (%f)" % (name, score))


# 讀檔
train_df, test_df = get_data('train'), get_data('test')

# 清理資料
x_train, x_valid, y_train, y_valid, x_test, cols, user_id = clean_data(train_df, test_df)
print('Columns:', cols)

# 建立ensemble空間
ensemble_test = pd.DataFrame()
ensemble_valid = pd.DataFrame()

# 預設分類器種類
classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    xgb.XGBClassifier()
]

# 建立分類器效度表
log_cols = ["Classifier", "F1"]
log = pd.DataFrame([], columns=log_cols)

# flags
gs = 0
smote = 1

# Catboost單獨跑...
# TODO:想看看能不能丟進classifiers
catboost_classifier(x_train, y_train, x_valid, y_valid, x_test, cols, hyper_tune=0, smote=0)
catboost_classifier(x_train, y_train, x_valid, y_valid, x_test, cols, hyper_tune=1, smote=0)

for clf in classifiers:
    # 先跑一次預設參數分類器
    log_entry = classifier(clf, 0, 0, x_train, y_train, x_valid, y_valid, x_test, cols)
    log = log.append(pd.DataFrame([log_entry], columns=log_cols))
    if gs:
        # gs設1則開始調參
        log_entry = classifier(clf, gs, 0, x_train, y_train, x_valid, y_valid, x_test, cols)
        log = log.append(pd.DataFrame([log_entry], columns=log_cols))
    print()

if smote:
    # 處理資料不平衡問題
    oversampling = BorderlineSMOTE(sampling_strategy=0.11, k_neighbors=10, n_jobs=-1,
                                   m_neighbors=10, kind='borderline-2')
    undersampling = RandomUnderSampler(sampling_strategy=0.3)
    steps = [('o', oversampling), ('u', undersampling)]
    pipeline = Pipeline(steps=steps)
    print()
    print('Imbalance Sampling...')
    print('Original dataset shape: %s' % Counter(y_train))
    x_train, y_train = pipeline.fit_resample(x_train, y_train)
    print('Resampled dataset shape: %s' % Counter(y_train))

    catboost_classifier(x_train, y_train, x_valid, y_valid, x_test, cols, hyper_tune=0, smote=smote)
    for clf in classifiers:
        log_entry = classifier(clf, 0, smote, x_train, y_train, x_valid, y_valid, x_test, cols)
        log = log.append(pd.DataFrame([log_entry], columns=log_cols))
        if gs:
            log_entry = classifier(clf, gs, smote, x_train, y_train, x_valid, y_valid, x_test, cols)
            log = log.append(pd.DataFrame([log_entry], columns=log_cols))
        print()


# ensemble
print(ensemble_valid.head())
ensemble_model = xgb.XGBClassifier()
ensemble_model.fit(ensemble_valid, y_valid.values.ravel())
predictions = ensemble_model.predict(ensemble_valid)

f_score = fbeta_score(y_valid, predictions, beta=1, average='binary')
log = log.append(pd.DataFrame([['Ensemble', f_score*100]], columns=log_cols))

importances = ensemble_model.feature_importances_
indices = np.argsort(importances)[::-1]
cols = ensemble_valid.columns
print("Feature ranking:")
for f in range(len(importances)):
    print("%3d. %25s  (%f)" % (f + 1, cols[indices[f]], importances[indices[f]]))

plot_result(y_valid, predictions)


# Visualize
sns.set_color_codes("muted")
g = sns.barplot(x='F1', y='Classifier', data=log, color="b")
plt.xlabel('F1 Score')
plt.title('Classifier\'s F1 Score')
for p in g.patches:
    x = p.get_x() + p.get_width() + .3
    y = p.get_y() + p.get_height()/2 + .1
    g.annotate("%.2f" % (p.get_width()), (x, y))

plt.savefig("output.png")
plt.show()


# submission
# submit_predictor = ensemble_model.predict(ensemble_test)
# submit = pd.DataFrame({'user_id': user_id, 'class': submit_predictor})
# submit.to_csv(r'fraud_test_.csv', index=False, header=True)


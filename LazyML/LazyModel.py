# -*- coding: utf-8 -*-
"""
Created on 2020/04/28 11:26:54

@author: 10356
"""
import lightgbm as lgb
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

__all__ = ["LazyClassifier",
           "LazyRegressor",
           ]


class LazyClassifier:
    def __init__(self, x_train, y_train, x_valid, y_valid, x_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test if x_test is not None else y_valid
        self.classes = len(set(y_train))
        if self.classes > 2:
            print('MultiClass. Set Scoring as "f1_macro"')
            self.scoring = 'f1_macro'
        else:
            self.scoring = 'f1'

        self.hyper_dict = {
            'lightGBM': {
                'space': {
                    'n_estimators': hp.quniform('n_estimators', 5, 200, 5),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
                    'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
                    'learning_rate': hp.loguniform('learning_rate', 1e-2, 5e-1),
                },
                'objective': self.bgm_objective,
                'trials': {},
            },
            'XGBoost': {
                'space': {
                    'n_estimators': hp.quniform('n_estimators', 5, 200, 5),
                    'learning_rate': hp.uniform('learning_rate', 1e-2, 5e-1),
                    'max_depth': hp.quniform('max_depth', 1, 12, 1),
                    'min_child_weight': hp.quniform('min_child_weight', 1, 12, 1),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
                    'gamma': hp.uniform('gamma', 0.0, 0.5),
                },
                'objective': self.xgb_objective,
                'trials': {},
            },
            'RandomForest': {
                'space': {
                    'n_estimators': hp.quniform('n_estimators', 5, 200, 5),
                    'max_depth': hp.quniform('max_depth', 1, 15, 1),
                    'max_features': hp.uniform('max_features', 0.0, 0.5),
                    "min_samples_split": hp.uniform('min_samples_split', 0.0, 0.5),
                    "min_samples_leaf": hp.quniform('min_samples_leaf', 1, 10, 1),
                },
                'objective': self.rf_objective,
                'trials': {},
            },
        }

    def rf_objective(self, params):
        params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'max_features': params['max_features'],
            "min_samples_split": params['min_samples_split'],
            "min_samples_leaf": int(params['min_samples_leaf']),
        }
        clf = RandomForestClassifier(n_jobs=8, class_weight='balanced', **params)
        score = cross_val_score(clf, self.x_train, self.y_train, scoring=self.scoring, cv=StratifiedKFold()).mean()
        print("F1: {:.3f} params {}".format(score, params))
        return -score

    def xgb_objective(self, params):
        params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'learning_rate': params['learning_rate'],
            'min_child_weight': int(params['min_child_weight']),
            'colsample_bytree': params['colsample_bytree'],
            'gamma': params['gamma'],
        }
        clf = xgb.XGBClassifier(n_jobs=8, **params)
        score = cross_val_score(clf, self.x_train, self.y_train, scoring=self.scoring, cv=StratifiedKFold()).mean()
        print("F1: {:.3f} params {}".format(score, params))
        return -score

    def bgm_objective(self, params):
        params = {
            'n_estimators': int(params['n_estimators']),
            'num_leaves': int(params['num_leaves']),
            'colsample_bytree': params['colsample_bytree'],
            'learning_rate': params['learning_rate'],
        }
        clf = lgb.LGBMClassifier(**params)
        score = cross_val_score(clf, self.x_train, self.y_train, scoring=self.scoring, cv=StratifiedKFold()).mean()
        print("F1: {:.5f} params {}".format(score, params))
        return -score

    def fit(self, model_name, hyper_space, hyper_objective, max_eval=20):
        trials = Trials()
        best = fmin(
            fn=hyper_objective,
            space=hyper_space,
            algo=tpe.suggest,
            max_evals=max_eval,
            trials=trials
        )
        print("{} Hyperopt estimated optimum {}".format(model_name, best))

        if model_name == 'lightGBM':
            best['n_estimators'] = int(best['n_estimators'])
            best['num_leaves'] = int(best['num_leaves'])
            hyper_model = lgb.LGBMClassifier(**best)
            hyper_model_ = lgb.LGBMClassifier()
        elif model_name == 'XGBoost':
            best['n_estimators'] = int(best['n_estimators'])
            best['max_depth'] = int(best['max_depth'])
            hyper_model = XGBClassifier(**best)
            hyper_model_ = XGBClassifier()
        elif model_name == 'RandomForest':
            best['n_estimators'] = int(best['n_estimators'])
            best['min_samples_leaf'] = int(best['min_samples_leaf'])
            hyper_model = RandomForestClassifier(**best)
            hyper_model_ = RandomForestClassifier()
        else:
            return 'Unknown Model: {}'.format(model_name)

        hyper_model.fit(self.x_train, self.y_train)
        hyper_model_.fit(self.x_train, self.y_train)
        predict_ = hyper_model.predict(self.x_valid)
        predict__ = hyper_model_.predict(self.x_valid)
        print('Default parameters.', '-' * 20)
        self.plot_result(self.y_valid, predict_)
        print('Hyperopt parameters.', '-' * 20)
        self.plot_result(self.y_valid, predict__)
        self.hyper_plot(hyper_space.keys(), trials, model_name)

    @staticmethod
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

    @staticmethod
    def hyper_plot(parameters, trials, name):
        f, axes = plt.subplots(nrows=1, ncols=len(parameters), figsize=(15, 5))
        plt.title('model_' + name)
        cmap = plt.cm.jet
        for i, val in enumerate(parameters):
            xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
            ys = [round(-t['result']['loss']*100, 1) for t in trials.trials]
            xs, ys = zip(*sorted(zip(xs, ys)))
            ys = np.array(ys)
            axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i) / len(parameters)))
            axes[i].set_title(val)
        plt.savefig('hyperopt_plt_' + name + '.png')


class LazyRegressor:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

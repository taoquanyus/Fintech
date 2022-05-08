"""
特征组合：Dict+GroupBy+nlp
特征选择方式：chi2
参数寻优办法：beyesian
模型：xgboost
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe
from scipy import sparse
from sklearn.feature_selection import f_regression
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def read_data(debug=True):
    """

    :param debug:
    :return:
    """
    print("read_data...")
    train = pd.read_csv("preprocesses/train_clean_encode.csv")
    test = pd.read_csv("preprocesses/test_clean_encode.csv")

    features = train.columns.tolist()
    features.remove('CUST_UID')
    features.remove('LABEL')

    train_x = sparse.csr_matrix(train[features])
    test_x = sparse.csr_matrix(test[features])
    print("done")
    return train_x, test_x


def params_append(params):
    """

    :param params:
    :return:
    """
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'auc'
    params["min_child_weight"] = int(params["min_child_weight"])
    params['max_depth'] = int(params['max_depth'])
    return params


def param_beyesian(train, xgb_cv=None):
    """

    :param xgb_cv:
    :param train:
    :return:
    """
    train_y = pd.read_csv("preprocesses/train_dict_encode.csv")['LABEL']
    sample_index = train_y.sample(frac=1, random_state=2020).index.tolist()
    train_data = xgb.DMatrix(train.tocsr()[sample_index, :
                             ], train_y.loc[sample_index].values, silent=True)
    def xgb_cv(colsample_bytree, subsample, min_child_weight, max_depth, eta):
        """
        :param colsample_bytree:
        :param subsample:
        :param min_child_weight:
        :param max_depth:
        :param reg_alpha:
        :param eta:
        :param reg_lambda:
        :return:
        """
        params = {'objective': 'binary:logistic',
                  'early_stopping_round': 25,
                  'eval_metric': 'auc'}
        params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['subsample'] = max(min(subsample, 1), 0)
        params["min_child_weight"] = int(min_child_weight)
        params['max_depth'] = int(max_depth)
        params['eta'] = float(eta)
        print(params)
        cv_result = xgb.cv(params, train_data,
                           num_boost_round=10000,
                           nfold=2, seed=2022,
                           stratified=False,
                           shuffle=True,
                           early_stopping_rounds=25,
                           verbose_eval=True)
        return max(cv_result['test-auc-mean'])
    xgb_bo = BayesianOptimization(
        xgb_cv,
        {'colsample_bytree': (0.5, 1),
         'subsample': (0.5, 1),
         'min_child_weight': (1, 30),
         'max_depth': (5, 12),
         'eta': (0.01, 0.5)
         }
    )
    xgb_bo.maximize(init_points=21, n_iter=5)  # init_points表示初始点，n_iter代表迭代次数（即采样数）
    print(xgb_bo.max['target'], xgb_bo.max['params'])
    return xgb_bo.max['params']


def train_predict(train, test, params):
    """

    :param train:
    :param test:
    :param params:
    :return:
    """
    train_y = pd.read_csv("preprocesses/train_clean.csv")['LABEL']
    test_data = xgb.DMatrix(test)

    params = params_append(params)
    kf = KFold(n_splits=5, random_state=2022, shuffle=True)
    prediction_test = 0
    cv_score = []
    prediction_train = pd.Series()
    ESR = 20
    NBR = 10000
    VBE = 50
    for train_part_index, eval_index in kf.split(train, train_y):
        # 模型训练
        train_part = xgb.DMatrix(train.tocsr()[train_part_index, :],
                                 train_y.loc[train_part_index])
        eval = xgb.DMatrix(train.tocsr()[eval_index, :],
                           train_y.loc[eval_index])
        bst = xgb.train(params, train_part, NBR, [(train_part, 'train'),
                                                          (eval, 'eval')], verbose_eval=VBE,
                        maximize=False, early_stopping_rounds=ESR, )
        prediction_test += bst.predict(test_data)
        eval_pre = bst.predict(eval)
        prediction_train = prediction_train.append(pd.Series(eval_pre, index=eval_index))
        score = np.sqrt(mean_squared_error(train_y.loc[eval_index].values, eval_pre))
        cv_score.append(score)
    print(cv_score, sum(cv_score) / 5)
    pd.Series(prediction_train.sort_index().values).to_csv("preprocesses/train_xgboost.csv", index=False)
    pd.Series(prediction_test / 5).to_csv("preprocesses/test_xgboost.csv", index=False)
    test = pd.read_csv('preprocesses/test_dict_encode.csv')
    test['LABEL'] = prediction_test / 5
    test[['CUST_UID', 'LABEL']].to_csv("result/submission_xgboost.txt", index=False, sep=' ', header=None)
    return

if __name__ == "__main__":
    train, test = read_data(debug=False)
    best_clf = param_beyesian(train)
    train_predict(train, test, best_clf)
# [3.6799306462307517, 3.6476521867457588, 3.698480976611057, 3.7718461304040853, 3.579301270046094] 3.6754422420075494
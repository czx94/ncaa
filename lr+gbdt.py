import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.metrics.ranking import roc_auc_score
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn import model_selection

from scipy.sparse.construct import hstack

from sklearn import model_selection

import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "./input"]).decode("utf8"))
def gbdt_lr_train(X_all, y_all):
    # 训练/测试数据分割
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.3, random_state = 42)

    # 定义GBDT模型
    gbdt = GradientBoostingClassifier(n_estimators=40, max_depth=3, verbose=0,max_features=0.5)

    # 训练学习
    gbdt.fit(X_train, y_train)

    # 预测及AUC评测
    y_pred_gbdt = gbdt.predict_proba(X_test)[:, 1]
    gbdt_auc = roc_auc_score(y_test, y_pred_gbdt)
    print('gbdt auc: %.5f' % gbdt_auc)

    # lr对原始特征样本模型训练
    lr = LogisticRegression()
    lr.fit(X_train, y_train)    # 预测及AUC评测
    y_pred_test = lr.predict_proba(X_test)[:, 1]
    lr_test_auc = roc_auc_score(y_test, y_pred_test)
    print('基于原有特征的LR AUC: %.5f' % lr_test_auc)

    # GBDT编码原有特征
    X_train_leaves = gbdt.apply(X_train)[:,:,0]
    X_test_leaves = gbdt.apply(X_test)[:,:,0]

    # 对所有特征进行ont-hot编码
    (train_rows, cols) = X_train_leaves.shape

    gbdtenc = OneHotEncoder()
    X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))

    # 定义LR模型
    lr = LogisticRegression()
    # lr对gbdt特征编码后的样本模型训练
    lr.fit(X_trans[:train_rows, :], y_train)
    # 预测及AUC评测
    y_pred_gbdtlr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
    gbdt_lr_auc1 = roc_auc_score(y_test, y_pred_gbdtlr1)
    print('基于GBDT特征编码后的LR AUC: %.5f' % gbdt_lr_auc1)

    # 定义LR模型
    lr = LogisticRegression(n_jobs=-1)
    # 组合特征
    X_train_ext = hstack([X_trans[:train_rows, :], X_train])
    X_test_ext = hstack([X_trans[train_rows:, :], X_test])

    print(X_train_ext.shape)
    # lr对组合特征的样本模型训练
    lr.fit(X_train_ext, y_train)

    # 预测及AUC评测
    y_pred_gbdtlr2 = lr.predict_proba(X_test_ext)[:, 1]
    gbdt_lr_auc2 = roc_auc_score(y_test, y_pred_gbdtlr2)
    print('基于组合特征的LR AUC: %.5f' % gbdt_lr_auc2)
#the seed information
df_seeds = pd.read_csv('./input/NCAATourneySeeds.csv')

#tour information
df_tour = pd.read_csv('./input/NCAATourneyCompactResults.csv')

df_seeds['seed_int'] = df_seeds['Seed'].apply( lambda x : int(x[1:3]) )

df_winseeds = df_seeds.loc[:, ['TeamID', 'Season', 'seed_int']].rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.loc[:, ['TeamID', 'Season', 'seed_int']].rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])

df_concat['DiffSeed'] = df_concat[['LSeed', 'WSeed']].apply(lambda x : 0 if x[0] == x[1] else 1, axis = 1)

#prepares sample submission
df_sample_sub = pd.read_csv('./input/SampleSubmissionStage1.csv')

df_sample_sub['Season'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[0]) )
df_sample_sub['TeamID1'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[1]) )
df_sample_sub['TeamID2'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[2]) )

winners = df_concat.rename( columns = { 'WTeamID' : 'TeamID1',
                                                       'LTeamID' : 'TeamID2',
                                                      'WScore' : 'Team1_Score',
                                                      'LScore' : 'Team2_Score'}).drop(['WSeed', 'LSeed', 'WLoc'], axis = 1)
winners['Result'] = 1.0

losers = df_concat.rename( columns = { 'WTeamID' : 'TeamID2',
                                                       'LTeamID' : 'TeamID1',
                                                      'WScore' : 'Team2_Score',
                                                      'LScore' : 'Team1_Score'}).drop(['WSeed', 'LSeed', 'WLoc'], axis = 1)

losers['Result'] = 0.0

train = pd.concat( [winners, losers], axis = 0).reset_index(drop = True)

train['Score_Ratio'] = train['Team1_Score'] / train['Team2_Score']
train['Score_Total'] = train['Team1_Score'] + train['Team2_Score']
train['Score_Pct'] = train['Team1_Score'] / train['Score_Total']

df_sample_sub['Season'].unique()

train_test_inner = pd.merge( train.loc[ train['Season'].isin([2014, 2015, 2016, 2017]), : ].reset_index(drop = True),
         df_sample_sub.drop(['ID', 'Pred'], axis = 1),
         on = ['Season', 'TeamID1', 'TeamID2'], how = 'inner' )

team1d_num_ot = train_test_inner.groupby(['Season', 'TeamID1'])['NumOT'].median().reset_index()\
.set_index('Season').rename(columns = {'NumOT' : 'NumOT1'})
team2d_num_ot = train_test_inner.groupby(['Season', 'TeamID2'])['NumOT'].median().reset_index()\
.set_index('Season').rename(columns = {'NumOT' : 'NumOT2'})

num_ot = team1d_num_ot.join(team2d_num_ot).reset_index()

#sum the number of ot calls and subtract by one to prevent overcounting
num_ot['NumOT'] = num_ot[['NumOT1', 'NumOT2']].apply(lambda x : round( x.sum() ), axis = 1 )

team1d_score_spread = train_test_inner.groupby(['Season', 'TeamID1'])[['Score_Ratio', 'Score_Pct']].median().reset_index()\
.set_index('Season').rename(columns = {'Score_Ratio' : 'Score_Ratio1', 'Score_Pct' : 'Score_Pct1'})
team2d_score_spread = train_test_inner.groupby(['Season', 'TeamID2'])[['Score_Ratio', 'Score_Pct']].median().reset_index()\
.set_index('Season').rename(columns = {'Score_Ratio' : 'Score_Ratio2', 'Score_Pct' : 'Score_Pct2'})

score_spread = team1d_score_spread.join(team2d_score_spread).reset_index()

#geometric mean of score ratio of team 1 and inverse of team 2
score_spread['Score_Ratio'] = score_spread[['Score_Ratio1', 'Score_Ratio2']].apply(lambda x : ( x[0] * ( x[1] ** -1.0) ), axis = 1 ) ** 0.5

#harmonic mean of score pct
score_spread['Score_Pct'] = score_spread[['Score_Pct1', 'Score_Pct2']].apply(lambda x : 0.5*( x[0] ** -1.0 ) + 0.5*( 1.0 - x[1] ) ** -1.0, axis = 1 ) ** -1.0

X_train = train_test_inner.loc[:, ['Season', 'NumOT', 'Score_Ratio', 'Score_Pct']]
train_labels = train_test_inner['Result']

train_test_outer = pd.merge( train.loc[ train['Season'].isin([2014, 2015, 2016, 2017]), : ].reset_index(drop = True),
         df_sample_sub.drop(['ID', 'Pred'], axis = 1),
         on = ['Season', 'TeamID1', 'TeamID2'], how = 'outer' )

train_test_outer = train_test_outer.loc[ train_test_outer['Result'].isnull(),
                                        ['TeamID1', 'TeamID2', 'Season']]

train_test_missing = pd.merge( pd.merge( score_spread.loc[:, ['TeamID1', 'TeamID2', 'Season', 'Score_Ratio', 'Score_Pct']],
                   train_test_outer, on = ['TeamID1', 'TeamID2', 'Season']),
         num_ot.loc[:, ['TeamID1', 'TeamID2', 'Season', 'NumOT']],
         on = ['TeamID1', 'TeamID2', 'Season'])

X_test = train_test_missing.loc[:, ['Season', 'NumOT', 'Score_Ratio', 'Score_Pct']]

n = X_train.shape[0]

train_test_merge = pd.concat( [X_train, X_test], axis = 0 ).reset_index(drop = True)

train_test_merge = pd.concat( [pd.get_dummies( train_test_merge['Season'].astype(object) ),
            train_test_merge.drop('Season', axis = 1) ], axis = 1 )

train_test_merge = pd.concat( [pd.get_dummies( train_test_merge['NumOT'].astype(object) ),
            train_test_merge.drop('NumOT', axis = 1) ], axis = 1 )

X_train = train_test_merge.loc[:(n - 1), :].reset_index(drop = True)
X_test = train_test_merge.loc[n:, :].reset_index(drop = True)

x_max = X_train.max()
x_min = X_train.min()

X_train = ( X_train - x_min ) / ( x_max - x_min + 1e-14)
X_test = ( X_test - x_min ) / ( x_max - x_min + 1)

#gbdt_lr_train(X_train, train_labels)
# 定义GBDT模型
gbdt_est = GradientBoostingClassifier(random_state=3, verbose=0)
gbdt_param_grid = {'n_estimators': [500, 1000], 'max_features':[0.001, 0.005], 'min_samples_split': [2], 'max_depth': [2, 3, 5]}
gbdt = model_selection.GridSearchCV(gbdt_est, gbdt_param_grid, n_jobs=25, cv=10, verbose=1)
gbdt.fit(X_train, train_labels)
print(gbdt.best_params_)
log_clf = gbdt
# GBDT编码原有特征
# X_train_leaves = gbdt.apply(X_train)[:,:,0]
# X_test_leaves = gbdt.apply(X_test)[:,:,0]
#
# # 对所有特征进行ont-hot编码
# (train_rows, cols) = X_train_leaves.shape
#
# gbdtenc = OneHotEncoder()
# X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))
#
# X_train_ext = hstack([X_trans[:train_rows, :], X_train])
# X_test_ext = hstack([X_trans[train_rows:, :], X_test])
#
# # lr
# log_clf = LogisticRegressionCV(cv = 10)
#
# log_clf.fit(X_train_ext, train_labels)
train_test_inner['Pred1'] = log_clf.predict_proba(X_train)[:,1]
train_test_missing['Pred1'] = log_clf.predict_proba(X_test)[:,1]
# #svm
# log_clf = svm.SVC(probability=True)
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100]}]
# clf = GridSearchCV(log_clf, tuned_parameters, cv=15,
#                        scoring='accuracy')
#     # 用训练集训练这个学习器 clf
# clf.fit(X_train, train_labels)
# #xgb
# xgb_est = xgb.XGBClassifier(learning_rate=0.03, random_state=3, n_estimators=900, subsample=0.8, n_jobs = 50,colsample_bytree = 0.8, verbose=1)
# xgb_param_grid = {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}
# xgb_grid = model_selection.GridSearchCV(xgb_est, xgb_param_grid, n_jobs=4, cv=20, verbose=1)
# xgb_grid.fit(X_train, train_labels)
# print('xgb:',xgb_grid.best_params_)
# clf = xgb_grid
# train_test_inner['Pred1'] = clf.predict_proba(X_train)[:,1]
# train_test_missing['Pred1'] = clf.predict_proba(X_test)[:,1]

sub = pd.merge(df_sample_sub,
                         pd.concat( [train_test_missing.loc[:, ['Season', 'TeamID1', 'TeamID2', 'Pred1']],
                                     train_test_inner.loc[:, ['Season', 'TeamID1', 'TeamID2', 'Pred1']] ],
                                   axis = 0).reset_index(drop = True),
                  on = ['Season', 'TeamID1', 'TeamID2'], how = 'outer')
team1_probs = sub.groupby('TeamID1')['Pred1'].apply(lambda x : (x ** -1.0).mean() ** -1.0 ).fillna(0.5).to_dict()
team2_probs = sub.groupby('TeamID2')['Pred1'].apply(lambda x : (x ** -1.0).mean() ** -1.0 ).fillna(0.5).to_dict()

sub['Pred'] = sub[['TeamID1', 'TeamID2','Pred1']]\
.apply(lambda x : team1_probs.get(x[0]) * ( 1 - team2_probs.get(x[1]) ) if np.isnan(x[2]) else x[2],
       axis = 1)

sub[['ID', 'Pred']].to_csv('sub.csv', index = False)
"""
Project: Titanic Survival Prediction
Dataset: Kaggle - Titanic: Machine Learning from Disaster
Description:
This project uses a Kaggle competition dataset to practicemachine learning techniques for titanic survival prediction.
The implementation is based on academic coursework and reference code, with modifications and adjustments made during the learning process.
"""

#查詢有無錯誤值
def na_check(df_data):
    data_na = (df_data.isnull().sum() / len(df_data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    display(missing_data.head(10))

#機器學習所需要的package
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()

#讀取訓練資料
import os
import numpy as np
import pandas as pd

data_path = '/home/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Ticket'] , axis=1)
df_test = df_test.drop(['PassengerId', 'Ticket'] , axis=1)

#看一下 資料的特徵
df_mix = pd.concat([df_train,df_test])
df_mix.describe()

# train 有無錯誤值
na_check(df_train)

# test 有無錯誤值
na_check(df_test)

# 寫成矩陣作為 之後修改特徵使用
combine = [df_train, df_test]
combine

for df in combine:
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# 把Name依ABC先後分類，先進行開頭的分類(例如:C的一組),然後第二個字母分類(例如:a的一組),以此類推,最後分類成title
# title 依男女進行分類,計算人數
pd.crosstab(df_train['Title'], df_train['Sex'])

for df in combine:
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    #將不常用尊稱歸為一類'Rare'
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    #Mlle:小姐，歸類為Miss(單身)
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    #Ms:女士(不明確為單身或結婚，歸類為單身，歸類為Miss(單身)
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    #Mme:太太,歸類為Mrs
for df in combine:
    df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    #將五種特徵標籤化
    df['Title'] = df['Title'].fillna(0)
    #NAN的寫0

df_train.head()

import re
deck = {"ABC": 1, "DE": 2, "FG": 3,  "U": 0}
#鐵達尼號分為不同的樓層，A為最高層,G為最底層,U為未知進行分類
for df in combine:
    df['Cabin'] = df['Cabin'].fillna("U0")
    df['Deck'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    df['Deck'] = df['Deck'].replace(['A','B','C'], 'ABC')
    df['Deck'] = df['Deck'].replace(['D','E'],'DE')
    df['Deck'] = df['Deck'].replace(['F','G'],'FG')
    df['Deck'] = df['Deck'].map(deck)
    df['Deck'] = df['Deck'].fillna(0)
    df['Deck'] = df['Deck'].astype(int)

for df in combine:
#男女分類
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} )

df_train.head()

for df in combine:
  #讓未知年紀隨機分配到平均正負一個標準差之間
    mean = df_train["Age"].mean()
    std = df_test["Age"].std()
    is_null = df["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    age_slice = df["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    df["Age"] = age_slice
    df["Age"] = df_train["Age"].astype(int)
df_train["Age"].isnull().sum()

#年紀最小是0歲、最大是80歲，切成5等分進行分類
for df in combine:
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age']
    df['Age'] = df['Age'].astype(int)
df_train.head()

#sibsp:兄弟姊妹,parch:小孩父母
#FamilySize= sibsp+parch+自己
for df in combine:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

for df in combine:
  #FamilySize==1只有一個人,所以是'IsAlone'
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

freq_port = df_train.Embarked.dropna().mode()[0]
for df in combine:
    df['Embarked'] = df['Embarked'].fillna(freq_port)
#freq_port:最多數量的Port of Embarkation

for df in combine:
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
#根據其標示做特徵分類

df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)
for df in combine:
  #根據特徵表述,25%是7.89,50%是14.454,75%是31.275,但離最大值有些距離,故再取99及250區間
    df.loc[ df['Fare'] <= 7.89, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.89) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31.275), 'Fare']   = 2
    df.loc[(df['Fare'] > 31.275 ) & (df['Fare'] <= 99), 'Fare']   = 3
    df.loc[(df['Fare'] > 99) & (df['Fare'] <= 250), 'Fare']   = 4
    df.loc[df['Fare'] > 250, 'Fare'] = 5
    df['Fare'] = df['Fare'].astype(int)

#已使用的特徵扔掉
df_train = df_train.drop(['Parch', 'SibSp', 'Name','FamilySize','Cabin'], axis=1)
df_test = df_test.drop(['Parch', 'SibSp', 'Name','FamilySize','Cabin'], axis=1)

df_train

#查看與生存的相關性,發現年紀的相關性低
corr_values_df = pd.DataFrame(df_train.corr()['Survived'].abs().sort_values(ascending=True))
corr_values_df

#丟掉低相關性的'Age'
df_train = df_train.drop(['Age'], axis=1)
df_test = df_test.drop(['Age'], axis=1)

df_train = df_train.drop(['Survived'], axis=1)

na_check(df_train)

param_grid = { "min_samples_leaf" : [1, 5, 10, 25], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100]}
from sklearn.model_selection import GridSearchCV, cross_val_score
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, random_state=1, n_jobs=-1)
clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
clf.fit(df_train, train_Y)
clf.best_params_

#根據上述,發現隨機森林最好的表示如下
rf = RandomForestClassifier(criterion = "gini",
                                       min_samples_leaf = 1,
                                        max_depth= 5,
                                       min_samples_split = 35,
                                       n_estimators=100,
                                       max_features='auto',
                                       oob_score=True,
                                       random_state=1,
                                       n_jobs=-1)

#預測檔
rf.fit(df_train, train_Y)
rf_pred = rf.predict(df_test)
sub = pd.DataFrame({'PassengerId': ids, 'Survived': rf_pred})
sub.to_csv('Titanic_pred3.csv', index=False)

#Project: House Price Prediction
#Dataset: Kaggle - House Prices: Advanced Regression Techniques

#Description:
#This project uses a Kaggle competition dataset to practice machine learning techniques for house price prediction.
#The implementation is based on academic coursework and reference code, with modifications and adjustments made during the learning process.


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
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()

#讀取訓練資料
import os
import numpy as np
import pandas as pd

data_path = '/home/'
df_train = pd.read_csv(data_path + 'house_train.csv.gz')
df_test = pd.read_csv(data_path + 'house_test.csv.gz')

train_Y = np.log1p(df_train['SalePrice'])
train_num = train_Y.shape[0]
ids = df_test['Id']
df_train = df_train.drop(['Id'] , axis=1)
df_test = df_test.drop(['Id'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

#讀取結果資料(sample_submission為kaggle在data所附的結果範例)
df_answer = pd.read_csv('/content/drive/MyDrive/人工智慧/data/sample_submission.csv')
answer = df_answer['SalePrice']
true = np.array(answer)
true

#'GrLivArea'在相同的'SaleCondition'下所做的平均、眾數、中位數、最大值
df['SaleCondition'] = df['SaleCondition'].fillna('None')
mean_df = df.groupby(['SaleCondition'])['GrLivArea'].mean().reset_index() #平均
mode_df = df.groupby(['SaleCondition'])['GrLivArea'].apply(lambda x: x.mode()[0]).reset_index() #眾數
median_df = df.groupby(['SaleCondition'])['GrLivArea'].median().reset_index() #中位數
max_df = df.groupby(['SaleCondition'])['GrLivArea'].max().reset_index() #最大值
temp = pd.merge(mean_df, mode_df, how='left', on=['SaleCondition'])
temp = pd.merge(temp, median_df, how='left', on=['SaleCondition'])
temp = pd.merge(temp, max_df, how='left', on=['SaleCondition']) #將平均、眾數、中位數、最大值一個一個組合
df = pd.merge(df, temp, how='left', on=['SaleCondition']) #將前四者與原本資料結合
df.shape

# 以下只是摘錄房產預測中的一組特徵工程, 並以此組特徵工程跑參數。
# 部分欄位缺值補 'None'
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
             'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass']
for col in none_cols:
    df[col] = df[col].fillna("None")

# Functional缺值補'typ'
df['Functional'] = df['Functional'].fillna("Typ")

# 部分欄位缺值填補 0
zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF',
             'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in zero_cols:
    df[col] = df[col].fillna(0)

# 部分欄位缺值補眾數
mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
for col in mode_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 'LotFrontage' 有空缺時, 以同一區 (Neighborhood) 的 LotFrontage 中位數填補
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# Utilities 參考資訊很少, 所以直接捨棄
df = df.drop(['Utilities'], axis=1)

# 四個數值欄位, 因為相異值有限, 轉成文字
label_cols = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']
for col in mode_cols:
    df[col] = df[col].astype(str)

# 相異值不太具有代表性的, 做標籤編碼
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(df[c].values))
    df[c] = lbl.transform(list(df[c].values))

# 由地下室面積 + 1樓面積 + 2樓面積, 計算總坪數特徵
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 把剩下少數重要的類別型欄位, 做獨熱編碼 (已變成數字的欄位, 會自動跳過)
df = pd.get_dummies(df)
print(df.shape)

#捨棄與SalePrice相關低的欄位
corr_values_df = pd.DataFrame(df.corr()['SalePrice'].abs().sort_values(ascending=True)) #排序相關性，第一項:相關性最低；最後一項:相關性最高
drop_features = corr_values_df.index.values[0:100]  #捨棄前100項(前100個低相關性)
df = df.drop(drop_features, axis = 1)
df.head()

#標準化 數值型欄位
standard = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'MasVnrArea', 'KitchenQual', 'LotFrontage']
for c in standard:
  df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1,1))
df=df.drop(['SalePrice'] , axis=1) #刪除SalePrice，因為前面相關性需要SalePrice，故最前面沒有刪除

#是否有NA值
na_check(df)

train_X = df[:train_num]
print(train_X.shape) # 確認train_X的size
test_X = df[train_num:]
print(test_X.shape) # 確認test_X的size
Lasso = Lasso(alpha = 0.0005, random_state=1)  #Lasso模型
reg = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5) #梯度提升機模型(參考他人在寫房價預測時所用的最佳模型)
linear = LinearRegression(normalize=False, fit_intercept=True, copy_X=True) #線性模型
rf = RandomForestRegressor(n_estimators=3000,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   random_state =5) #隨機森林模型(參考他人在寫房價預測時所用的最佳模型)

linear.fit(train_X, train_Y)
linear_pred = linear.predict(test_X) #線性預測
reg.fit(train_X, train_Y)
reg_pred = reg.predict(test_X) #梯度提升機預測
rf.fit(train_X, train_Y)
rf_pred = rf.predict(test_X) #隨機森林預測
Lasso.fit(train_X, train_Y)
Lasso_pred = rf.predict(test_X) # Lasso預測
blending_pred = linear_pred*0.30 + reg_pred*0.67 + Lasso_pred*0.03  #混合泛化
sub = pd.DataFrame({'Id': ids, 'SalePrice': np.expm1(blending_pred)})
sub.to_csv('house_pred.csv', index=False)

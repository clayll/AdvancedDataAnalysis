import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import missingno as msno

'''
## 目标是根据人口普查核实数据
### 查看数据概览
### 单变量分析
### 双变量分析
### 特征工程,数据清洗
* 结果标签观察
* 特征清洗分析
* 特征合并
* 特征构建
### 模型建立
'''
headers = ['age', 'workclass', 'fnlwgt',
           'education', 'education-num',
           'marital-status', 'occupation',
           'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country',
           'predclass']
training_raw = pd.read_csv("dataset/adult.data",names=headers,header=None)
test_raw = pd.read_csv("dataset/adult.test",names=headers,header=None)
print(training_raw.columns)

# 合并进行分析

test_raw.drop(axis=0,index=0,inplace=True)
dataset_raw = training_raw.append(test_raw,ignore_index=True)
dataset_raw.age = dataset_raw.age.apply(np.int)
# 新建两个集合，分别来计算把连续变量离散化和未离散化的数据
# 分别对各个数据中的字段进行合并，并进行整理
# 离散变量数据集
dataset_bin = pd.DataFrame()
# 连续变量数据集
dataset_con = pd.DataFrame()

# 处理标签lable
dataset_raw.loc[dataset_raw.predclass == ' <=50K','predclass'] = 0
dataset_raw.loc[dataset_raw.predclass == ' <=50K.','predclass'] = 0
dataset_raw.loc[dataset_raw.predclass == ' >50K','predclass'] = 1
dataset_raw.loc[dataset_raw.predclass == ' >50K.','predclass'] = 1

dataset_bin['predclass'] = dataset_raw['predclass']
dataset_con['predclass'] = dataset_raw['predclass']
#
# g = sns.countplot(y ='predclass',data=dataset_raw)
# g.set_title('predclass')


# 处理age列，分为连续变量和随机变量
dataset_bin['age'] = pd.cut(x=dataset_raw.age,bins=10)
dataset_con['age'] = dataset_raw.age

print(dataset_bin.head(10))
# plt.figure(figsize=(10,5))
# plt.subplot(121)
# sns.countplot(y='age',data=dataset_bin)
# plt.subplot(122)
# sns.distplot(dataset_con.loc[dataset_con.predclass==0,'age'], kde_kws={"label": "<$50K"})
# sns.distplot(dataset_con.loc[dataset_con.predclass==1,'age'], kde_kws={"label": ">=$50K"})
# plt.show()


# 标签处理
def plot_distribution(dataset :pd.DataFrame,cols=3,width=40,height=15):
    rows = math.ceil(dataset.shape[1]/cols)
    plt.figure(figsize=(width,height))
    for index,column in enumerate(dataset.columns):
        plt.subplot(rows,cols,index+1)
        plt.xticks(rotation=25)
        if dataset[column].dtype == np.object:
            g = sns.countplot(y=column,data=training_raw)
            g.set_title(column)
            g.set_xlabel("数量")
            g.set_yticklabels([i.get_text() if len(i.get_text()) < 12 else i.get_text()[0:12]+'..' for i  in g.get_yticklabels()])
        else:
            sns.distplot(training_raw[column])


def plot_distribution(dataset :pd.DataFrame,cols=3,width=40,height=15):
    rows = math.ceil(dataset.shape[1]/cols)
    plt.figure(figsize=(width,height))
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    for index,column in enumerate(dataset.columns):
        plt.subplot(rows,cols,index+1)
        plt.xticks(rotation=25)
        if dataset[column].dtype == np.object:
            g = sns.countplot(y=column,data=training_raw)
            g.set_title(column)
            g.set_xlabel("数量")
            g.set_yticklabels([i if len(i) < 12 else i[0:12]+'..' for i  in dataset[column].values])
        else:
            sns.distplot(training_raw[column])

# plot_distribution(dataset=training_raw)
# plt.show()
# msno.matrix(df=training_raw)
# plt.show()
print (pd.cut(dataset_raw.age,10))
# plt.figure(figsize=(10,5))
# plt.subplot(121)
# sns.countplot(data=dataset_con,y='age',hue='predclass')
# plt.subplot(122)
# sns.distplot(dataset_con.loc[dataset_con.predclass==1,'age'],kde_kws={"label":">50k"})
# sns.distplot(dataset_con.loc[dataset_con.predclass==0,'age'],kde_kws={"label": "<=50K"})


def plot_bivariate_bar(dataset :pd.DataFrame,hue,cols=3,width=20,height=15):
    rows = math.ceil(dataset.shape[1]/cols)
    plt.figure(figsize=(width,height))
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    dataset.select_dtypes(include='object')
    for index,column in enumerate(dataset.columns):
        plt.subplot(rows,cols,index+1)
        plt.xticks(rotation=25)
        if np.object == dataset[column].dtype:
            g = sns.countplot(y=column,data=dataset,hue=hue)
            g.set_title(column)
            dataset['age'].astype()
        # g.set_xlabel("数量")
        # g.set_yticklabels([i if len(i) < 12 else i[0:12]+'..' for i  in dataset[column].values])

# sns.boxplot(data=dataset_raw,hue='predclass',y='education-num',x = 'marital-status')
# g = sns.FacetGrid(data=dataset_raw,col='marital-status', size=4, aspect=.7)
# g.map(sns.boxplot,'predclass','education-num')

# plt.figure(figsize=(20,15))
# plt.subplot(131)
# sns.violinplot(data=dataset_raw,x='predclass',y='education-num',hue='sex',palette="muted",split=True,scale='count')
# plt.subplot(132)
# sns.violinplot(data=dataset_raw,x='predclass',y='age',hue='sex',palette="muted",split=True,scale='count')
# plt.subplot(133)
# sns.violinplot(data=dataset_raw,x='predclass',y='hours-per-week',hue='sex',palette="muted",split=True,scale='count')

plt.figure(figsize=(20,30))
print(dataset_raw.dtypes['predclass'])
# 不同特征之间的散点图分布
# g = sns.pairplot(data= dataset_raw[['age','education-num','hours-per-week','capital-gain','capital-loss','predclass']],
#              hue='predclass',vars=dataset_raw[['age','education-num','hours-per-week','capital-gain','capital-loss','predclass']].columns[:-1],
#              size=4);
# # g.map_diag(sns.distplot)
# g.map_offdiag(sns.scatterplot)



# dataset_raw['age-hours'] = dataset_raw['age']*dataset_raw['hours-per-week']
# # dataset_bin['age-hours'] = pd.cut(dataset_raw['age-hours'],10)
# # dataset_con['age-hours'] = dataset_raw['age-hours']
# #
# # plt.figure(figsize=(20,5))
# # plt.subplot(121)
# # sns.countplot(data=dataset_bin,y='age-hours')
# # plt.subplot(122)
# # sns.distplot(dataset_con['age-hours'])
# # plt.show()

# 性别和婚姻
dataset_raw['sex-marital']  = dataset_raw['sex'] + dataset_raw['marital-status']
dataset_con['sex-marital']  = dataset_raw['sex-marital']
dataset_bin['sex-marital']  = dataset_raw['sex-marital']
print(dataset_bin['sex-marital'].unique())
# plt.style.use('seaborn-whitegrid')
# fig = plt.figure(figsize=(20,5))
# sns.countplot(y="sex-marital", data=dataset_bin);

# test_pd = pd.DataFrame(data={'L1':[1,2,3],'L2':['a','b','c']})
# test_pd['L1'] = test_pd['L1'].astype(np.object)
# print(pd.get_dummies(test_pd))
#
# print(test_pd['L2'].factorize())
#
# dataset_bin_columns = dataset_bin.columns.tolist()
# dataset_bin_columns.remove('predclass')
# print(dataset_bin_columns)
# dataset_bin_columns = pd.get_dummies(dataset_bin,columns=dataset_bin_columns)
# print(dataset_bin)

a = [[1,4],[3,2]]
print(np.cov(a))

# a=np.array([1,2,3])
# b=np.array([1,2,4])
# x=np.vstack((a,b))
# print(x)#打印x的值
# print(np.cov(a,b))#计算协方差矩阵
# corr_coef = np.corrcoef(a,b)[0,1]#这里取得是第 0行 第2列的元素，为两者相关系数
# print(corr_coef)

#
# a=np.array([-1,-1,0,2,0])
# b=np.array([-2,0,0,1,1])
# x=np.vstack((a,b))
# print(x)#打印x的值
# print(np.cov(a,b))#计算协方差矩阵
# corr_coef = np.corrcoef(a,b)[0,1]#这里取得是第 0行 第2列的元素，为两者相关系数
# print(corr_coef)
from sklearn.preprocessing import LabelEncoder
print(dataset_raw.corr())


dataset_con_test = dataset_raw

mask = np.zeros_like(dataset_raw.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(data=dataset_raw.corr(),cmap=sns.color_palette("RdBu_r", 100),square=True)
# plt.show()

# dataset_con_test['marital-status'] = dataset_con['marital-status'].factorize()[0]


# dataset_con_enc = dataset_con_test.apply(LabelEncoder().fit_transform)
#
# print(dataset_con_enc.head())

# -*- coding: utf-8 -*-
# from sklearn.preprocessing import LabelEncoder
# # le = LabelEncoder().fit(list("abcde"))
# # le_transform = le.transform(['a','b','c'])
#
# te = pd.DataFrame(data={"l1":[1,2,3],"l2":['a','b','c']})
# le = te.apply(LabelEncoder().fit_transform)
# print(le)
# # print(te)
#
# print(dataset_con.describe(include='all'))
from sklearn.ensemble import RandomForestClassifier
data_set = dataset_raw
data_set = data_set.apply(LabelEncoder().fit_transform)
r = RandomForestClassifier()
r.fit(X=data_set.drop('predclass',axis=1),y=data_set.predclass)
importance = (r.feature_importances_)

rs = pd.DataFrame(data=importance,index=data_set.drop('predclass',axis=1).columns,columns=['importance'])
print(rs)
rs.sort_values(by='importance').plot(kind='barh')

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

scaler = StandardScaler().fit(X=data_set.drop('predclass',axis=1))
X = scaler.transform(X=data_set.drop('predclass',axis=1))
fit1  = PCA(n_components= len(data_set.columns)-1)

# plt.figure(figsize=(20,5))
# plt.subplot(121)
# # plt.bar(range(0,fit1.explained_variance_ratio_.size),fit1.explained_variance_ratio_)
#
#
# # Formatting
# target_names = [0,1]
# colors = ['navy','darkorange']
# lw = 2
# alpha = 0.3
#
# z = zip(colors, [0, 1], target_names)
# print(list(z))
#
# ax = plt.subplot(1, 2, 2, projection='3d')

from sklearn.feature_selection import RFECV
from sklearn.linear_model import  LogisticRegression
# selector1 = RFECV(LogisticRegression(), step=1, cv=5, n_jobs=-1)
# # selector1 = selector1.fit(data_set.drop('predclass', axis=1).values, data_set['predclass'].values)
# # print("Feature Ranking For Non-Discretised: %s" % selector1.ranking_)
# # print("Optimal number of features : %d" % selector1.n_features_)
# # print(" The mask of selected features:%s" % selector1.support_)
# # print(" The mask of selected features:%s" % selector1.support_)
# # print(np.insert(selector1.support_, 0, True))
lr = LogisticRegression()
from sklearn.model_selection import RandomizedSearchCV
param_dist = {'penalty': ['l2', 'l1'],
                         'class_weight': [None, 'balanced'],
                         'C': np.logspace(-20, 20, 10000),
                         'intercept_scaling': np.logspace(-20, 20, 10000)}
RandomizedSearchCV(lr,param_distributions=param_dist,n_jobs=-1)

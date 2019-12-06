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

g = sns.countplot(y ='predclass',data=dataset_raw)
g.set_title('predclass')


# 处理age列，分为连续变量和随机变量
dataset_bin['age'] = pd.cut(x=dataset_raw.age,bins=10)
dataset_con['age'] = dataset_raw.age

print(dataset_bin.head(10))
plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(y='age',data=dataset_bin)
plt.subplot(122)
sns.distplot(dataset_con.loc[dataset_con.predclass==0,'age'], kde_kws={"label": "<$50K"})
sns.distplot(dataset_con.loc[dataset_con.predclass==1,'age'], kde_kws={"label": ">=$50K"})
plt.show()


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


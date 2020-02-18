import  pandas as pd
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report,confusion_matrix

# pandas 读取数据：
df = pd.read_csv(r'C:\\百度网盘下载\\17.使用word2vec分类任务\\word2vec\\kaggle-word2vec\\data\\labeledTrainData.tsv',sep='\t',escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
print(df.head())

#去停用词
stopwords = {}.fromkeys([ line.rstrip() for line in open('stopwords.txt')])
eng_stopwords = set(stopwords)


# 对影评数据做预处理，大概有以下环节：
# # 去掉html标签
# # 移除标点
# # 切分成词/token
# # 去掉停用词
# # 重组为新的句子
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)

# 清洗每列数据
df['clean_review'] = df['review'].apply(clean_text)

vectorizer = CountVectorizer(max_features = 5000)
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()
print(train_data_features.shape)

X_train, X_test, y_train, y_test = train_test_split(train_data_features,df['sentiment'],test_size=0.2,random_state=0)

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    cm:confusion_matrix返回的结果
    classes：标签名称
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches


def plot_LSA(test_data, test_labels,  plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
        color_column = [color_mapper[label] for label in test_labels]
        colors = ['orange','blue']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
            red_patch = mpatches.Patch(color='orange', label='0')
            green_patch = mpatches.Patch(color='blue', label='1')
            plt.legend(handles=[red_patch, green_patch], prop={'size': 30})


fig = plt.figure(figsize=(16, 16))
plot_LSA(X_train, y_train)
plt.show()

log = LogisticRegression()
log.fit(X_train,y_train)
y_predict = log.predict(X_test)
print(classification_report(y_test,y_predict))
cm = confusion_matrix(y_test,y_predict)
# svd = truncated_svd(n_components=2, n_iter=7, random_state=42)
# cm = svd.fit(X_train)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm,class_names)


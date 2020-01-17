import pandas as pd
import numpy as np

import matplotlib.pyplot as plt; plt.rcdefaults()

#把数据读入

# 如果想更详细的了解数据的情况，可以打印其info信息，来观察不同列的类型以及整体占用内存，
# 这里教大家一个比较实用的技巧，如果拿到的数据非常大，对数据进行处理的时候可能会出现内存溢出的错误，
# 这里最简单的方法就是设置下数据个格式，比如将float64用float32来替代，这样可以大大节省内存开销。


##最受欢迎的一首歌曲有726885次播放。 刚才大家也看到了，这个音乐数据量集十分庞大，考虑到执行过程的时间消耗以及矩阵稀疏性问题，
# 我们依据播放量指标对数据集进行了截取。因为有些注册用户可能只是关注了一下之后就不再登录平台，这些用户对我们建模不会起促进作用，
# 反而增大了矩阵的稀疏性。对于歌曲也是同理，可能有些歌曲根本无人问津。由于之前已经对用户与歌曲播放情况进行了排序，
# 所以我们分别选择了其中的10W名用户和3W首歌曲，关于截取的合适比例大家也可以通过观察选择数据的播放量占总体的比例来设置。
## 歌曲播放次数
song_playcount = pd.read_csv(r"C:\Users\刘靓\Desktop\推荐数据\song_playcount_df.csv")
print(song_playcount.sort_values(by='play_count',ascending=False).head())

## 用户听歌次数
user_playcount = pd.read_csv(r"C:\Users\刘靓\Desktop\推荐数据\user_playcount_df.csv")
print(user_playcount.head())


## 歌曲基本信息
track_metadata = pd.read_csv(r"C:\Users\刘靓\Desktop\推荐数据\track_metadata_df_sub.csv",encoding='gbk')
print(track_metadata.columns)

## 清洗数据集
### 去除掉无用的和重复的，数据清洗是很重要的一步
def qingxi(triplet_dataset_sub_song_merged):
    triplet_dataset_sub_song_merged.rename(columns={'play_count': 'listen_count'}, inplace=True)
    print(triplet_dataset_sub_song_merged.columns)
    # 去掉不需要的指标
    # del (triplet_dataset_sub_song_merged['song_id'])
    triplet_dataset_sub_song_merged.drop(axis=1, labels='song_id', inplace=True)
    del (triplet_dataset_sub_song_merged['artist_id'])
    del (triplet_dataset_sub_song_merged['duration'])
    del (triplet_dataset_sub_song_merged['artist_familiarity'])
    del (triplet_dataset_sub_song_merged['artist_hotttnesss'])
    del (triplet_dataset_sub_song_merged['track_7digitalid'])
    del (triplet_dataset_sub_song_merged['shs_perf'])
    del (triplet_dataset_sub_song_merged['shs_work'])


'''合并数据'''
def mergeData():
    return track_metadata.merge(right=song_playcount,how='right',left_on= 'song_id',right_on='song')


triplet_dataset_sub_song_merged = mergeData()

qingxi(triplet_dataset_sub_song_merged)
print(triplet_dataset_sub_song_merged.columns)
print(triplet_dataset_sub_song_merged.head())
# triplet_dataset_sub_song_merged.sort_values(by='listen_count',ascending=False)


#展示最流行的歌曲
plt.subplot(221)
popsong = triplet_dataset_sub_song_merged[['title','listen_count']].sort_values(by='listen_count',ascending=False).reset_index()
plt.bar(x=popsong.loc[0:20,'title'],height=popsong.loc[0:20,'listen_count'])
plt.xticks( rotation='vertical')
# 最受欢迎的releases
plt.subplot(222)
popreleases = triplet_dataset_sub_song_merged.groupby(by='release').sum().sort_values(by='listen_count',ascending=False).head(20).reset_index()
plt.bar(x=popreleases['release'],height=popreleases['listen_count'])
plt.xticks(rotation='vertical')

plt.subplot(223)
# 最受欢迎的歌手
popartist = triplet_dataset_sub_song_merged.groupby(by='artist_name').sum().sort_values(by='listen_count',ascending=False).head(20).reset_index()
plt.bar(x=popartist['artist_name'],height=popartist['listen_count'])
plt.xticks(rotation='vertical')
plt.subplot(224)

popuser = user_playcount.groupby(by='user').sum().sort_values(by='play_count',ascending=False).reset_index()
plt.hist(x=popuser['play_count'],bins=50)
plt.grid(True)
plt.show()



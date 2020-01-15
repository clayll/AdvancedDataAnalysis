import pandas as pd


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
print(track_metadata[track_metadata.song_id=='SOBONKR12A58A7A7E0'])

## 清洗数据集
### 去除掉无用的和重复的，数据清洗是很重要的一步
def qingxi():
    print(song_playcount.shape)
    print( user_playcount.shape)
    print(track_metadata.shape)

qingxi()
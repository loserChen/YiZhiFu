# coding: utf-8
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import gc
import re
from lightgbm.sklearn import LGBMClassifier
from gensim.models import Word2Vec
import pickle
from scipy.stats import entropy, kurtosis

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('float_format', lambda x: '%.6f' % x)
pd.options.display.max_rows = None


def load_dataset(DATA_PATH):
    train_label = pd.read_csv(DATA_PATH+'train_label.csv')
    train_base = pd.read_csv(DATA_PATH+'train_base.csv')
    test_base = pd.read_csv(DATA_PATH+'testb_base.csv')

    train_op = pd.read_csv(DATA_PATH+'train_op.csv')
    train_trans = pd.read_csv(DATA_PATH+'train_trans.csv')
    test_op = pd.read_csv(DATA_PATH+'testb_op.csv')
    test_trans = pd.read_csv(DATA_PATH+'testb_trans.csv')

    return train_label, train_base, test_base, train_op, train_trans, test_op, test_trans

def transform_time(x):
    day = int(x.split(' ')[0])
    hour = int(x.split(' ')[2].split('.')[0].split(':')[0])
    minute = int(x.split(' ')[2].split('.')[0].split(':')[1])
    second = int(x.split(' ')[2].split('.')[0].split(':')[2])
    return 86400*day+3600*hour+60*minute+second

def data_preprocess(DATA_PATH):
    train_label, train_base, test_base, train_op, train_trans, test_op, test_trans = load_dataset(DATA_PATH=DATA_PATH)
    # 拼接数据
    train_df = train_base.copy()
    test_df = test_base.copy()
    train_df = train_label.merge(train_df, on=['user'], how='left')
    del train_base, test_base

    op_df = pd.concat([train_op, test_op], axis=0, ignore_index=True)
    trans_df = pd.concat([train_trans, test_trans], axis=0, ignore_index=True)
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    del train_op, test_op, train_df, test_df
    # 时间维度的处理
    op_df['days_diff'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[0]))
    trans_df['days_diff'] = trans_df['tm_diff'].apply(lambda x: int(x.split(' ')[0]))
    op_df['timestamp'] = op_df['tm_diff'].apply(lambda x: transform_time(x))
    trans_df['timestamp'] = trans_df['tm_diff'].apply(lambda x: transform_time(x))
    op_df['hour'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[0]))
    trans_df['hour'] = trans_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[0]))
    trans_df['minute'] = trans_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[1]))
    op_df['minute'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[1]))
    trans_df['week'] = trans_df['days_diff'].apply(lambda x: x % 7)
    op_df['week'] = op_df['days_diff'].apply(lambda x: x%7)
    trans_df['half_month'] = trans_df['days_diff'].apply(lambda x: x % 15)
    op_df['half_month'] = op_df['days_diff'].apply(lambda x: x % 15)
    # 排序
    trans_df = trans_df.sort_values(by=['user', 'timestamp'])
    op_df = op_df.sort_values(by=['user', 'timestamp'])
    trans_df.reset_index(inplace=True, drop=True)
    op_df.reset_index(inplace=True, drop=True)

    gc.collect()
    return data, op_df, trans_df

def gen_user_amount_features(df):
    group_df = df.groupby(['user'])['amount'].agg({
        'user_amount_mean': 'mean',
        'user_amount_std': 'std',
        'user_amount_max': 'max',
        'user_amount_min': 'min',
        'user_amount_sum': 'sum',
        'user_amount_med': 'median',
        'user_amount_cnt': 'count',
        'user_amount_skew': 'skew',
        'user_ampunt_nunique': 'nunique'
        }).reset_index()
    return group_df

def gen_user_group_amount_features(df, value):
    group_df = df.pivot_table(index='user',
                              columns=value,
                              values='amount',
                              dropna=False,
                              aggfunc=['count', 'sum']).fillna(0)
    group_df.columns = ['user_{}_{}_amount_{}'.format(value, f[1], f[0]) for f in group_df.columns]
    group_df.reset_index(inplace=True)

    return group_df

def gen_user_window_amount_features(df, window):
    group_df = df[df['days_diff']>window].groupby('user')['amount'].agg({
        'user_amount_mean_{}d'.format(window): 'mean',
        'user_amount_std_{}d'.format(window): 'std',
        'user_amount_max_{}d'.format(window): 'max',
        'user_amount_min_{}d'.format(window): 'min',
        'user_amount_sum_{}d'.format(window): 'sum',
        'user_amount_med_{}d'.format(window): 'median',
        'user_amount_cnt_{}d'.format(window): 'count',
        'user_amount_skew_{}d'.format(window): 'skew',
        'user_amount_nun_{}d'.format(window): 'nunique'
        }).reset_index()
    return group_df

def gen_user_window_amount_features_hour(df, window):
    group_df = df[(df['hour']>=window)&(df['hour']<=window+9)].groupby('user')['amount'].agg({
        'user_amount_mean_{}d'.format(window): 'mean',
        'user_amount_std_{}d'.format(window): 'std',
        'user_amount_max_{}d'.format(window): 'max',
        'user_amount_min_{}d'.format(window): 'min',
        'user_amount_sum_{}d'.format(window): 'sum',
        'user_amount_med_{}d'.format(window): 'median',
        'user_amount_cnt_{}d'.format(window): 'count',
        'user_amount_skew_{}d'.format(window): 'skew',
        'user_amount_nun_{}d'.format(window): 'nunique'
        }).reset_index()
    return group_df

def get_time_frequence(df,other,prefix):
    # 每一个用户每一天每一小时操作数的最大值，最小值，均值，标准差
    frequence_one_hour = other[['user', 'days_diff','hour']]
    frequence_one_hour['everyday_everyday_everyhour'] = 1
    frequence_one_hour = frequence_one_hour.groupby(['user','days_diff', 'hour']).agg('count').reset_index()
    frequence_one_hour = frequence_one_hour.drop(['days_diff', 'hour'], axis=1)
    frequence_one_hour = frequence_one_hour.groupby('user')['everyday_everyday_everyhour'].agg(
        {'mean', 'max', 'min', 'std'}).reset_index()
    frequence_one_hour.columns = ['user',
                                  'frequence_one_hour_mean_{}'.format(prefix),
                                  'frequence_one_hour_max_{}'.format(prefix),
                                  'frequence_one_hour_min_{}'.format(prefix),
                                  'frequence_one_hour_std_{}'.format(prefix)]
    df = pd.merge(df, frequence_one_hour, on='user', how='left')
    del frequence_one_hour
    # 每一个用户每一天操作数的最大值，最小值，均值，标准差
    frequence_one_day = other[['user','days_diff']]
    frequence_one_day['everyday'] = 1
    frequence_one_day = frequence_one_day.groupby(['user','days_diff']).agg('count').reset_index()
    frequence_one_day = frequence_one_day.drop(['days_diff'], axis=1)
    frequence_one_day = frequence_one_day.groupby('user')['everyday'].agg({'mean', 'max', 'min', 'std'}).reset_index()
    frequence_one_day.columns = ['user',
                                 'frequence_one_day_mean_{}'.format(prefix),
                                 'frequence_one_day_max_{}'.format(prefix),
                                 'frequence_one_day_min_{}'.format(prefix),
                                 'frequence_one_day_std_{}'.format(prefix)]
    df = pd.merge(df, frequence_one_day, on='user', how='left')
    del frequence_one_day

    # 每一个用户每一周操作数的最大值，最小值，均值，标准差
    frequence_one_week = other[['user','week']]
    frequence_one_week['everyweek'] = 1
    frequence_one_week = frequence_one_week.groupby(['user', 'week']).agg('count').reset_index()
    frequence_one_week = frequence_one_week.drop(['week'], axis=1)
    frequence_one_week = frequence_one_week.groupby('user')['everyweek'].agg(
        {'mean', 'max', 'min', 'std'}).reset_index()
    frequence_one_week.columns = ['user',
                                  'frequence_one_week_mean_{}'.format(prefix),
                                  'frequence_one_week_max_{}'.format(prefix),
                                  'frequence_one_week_min_{}'.format(prefix),
                                  'frequence_one_week_std_{}'.format(prefix)]
    df = pd.merge(df, frequence_one_week, on='user', how='left')
    del frequence_one_week

    # 每一个用户每一天半夜、早上、下午、晚上的最大值，最小值，均值，标准差
    frequence_one_hour = other[['user', 'days_diff', 'hour']]
    frequence_one_hour['everyday_everyhour'] = 1
    frequence_one_hour = frequence_one_hour.groupby(['user', 'days_diff', 'hour']).agg('count').reset_index()

    frequence_morning = frequence_one_hour[frequence_one_hour.hour <= 5][['user', 'days_diff', 'everyday_everyhour']]
    frequence_morning['everyday_morning'] = frequence_morning['everyday_everyhour']
    frequence_morning = frequence_morning.groupby(['user', 'days_diff'])['everyday_morning'].agg('sum').reset_index()
    frequence_morning = frequence_morning[['user', 'everyday_morning']]
    frequence_morning = frequence_morning.groupby('user')['everyday_morning'].agg(
        {'mean', 'max', 'min', 'std'}).reset_index()
    frequence_morning.columns = ['user',
                                 'frequence_morning_mean_{}'.format(prefix),
                                 'frequence_morning_max_{}'.format(prefix),
                                 'frequence_morning_min_{}'.format(prefix),
                                 'frequence_morning_std_{}'.format(prefix)]
    df = pd.merge(df, frequence_morning, on='user', how='left')
    del frequence_morning

    frequence_work_time = frequence_one_hour[(frequence_one_hour.hour <= 11) & (frequence_one_hour.hour >= 6)][
        ['user', 'days_diff', 'everyday_everyhour']]
    frequence_work_time['everyday_work_time'] = frequence_work_time['everyday_everyhour']
    frequence_work_time = frequence_work_time.groupby(['user', 'days_diff'])['everyday_work_time'].agg('sum').reset_index()
    frequence_work_time = frequence_work_time[['user', 'everyday_work_time']]
    frequence_work_time = frequence_work_time.groupby('user')['everyday_work_time'].agg(
        {'mean', 'max', 'min', 'std'}).reset_index()
    frequence_work_time.columns = ['user',
                                   'frequence_work_time_mean_{}'.format(prefix),
                                   'frequence_work_time_max_{}'.format(prefix),
                                   'frequence_work_time_min_{}'.format(prefix),
                                   'frequence_work_time_std_{}'.format(prefix)]
    df = pd.merge(df, frequence_work_time, on='user', how='left')
    del frequence_work_time

    frequence_afternoon = frequence_one_hour[(frequence_one_hour.hour <= 17) & (frequence_one_hour.hour >= 12)][
        ['user', 'days_diff', 'everyday_everyhour']]
    frequence_afternoon['everyday_afternoon'] = frequence_afternoon['everyday_everyhour']
    frequence_afternoon = frequence_afternoon.groupby(['user', 'days_diff'])['everyday_afternoon'].agg('sum').reset_index()
    frequence_afternoon = frequence_afternoon[['user', 'everyday_afternoon']]
    frequence_afternoon = frequence_afternoon.groupby('user')['everyday_afternoon'].agg(
        {'mean', 'max', 'min', 'std'}).reset_index()
    frequence_afternoon.columns = ['user',
                                   'frequence_afternoon_mean_{}'.format(prefix),
                                   'frequence_afternoon_max_{}'.format(prefix),
                                   'frequence_afternoon_min_{}'.format(prefix),
                                   'frequence_afternoon_std_{}'.format(prefix)]
    df = pd.merge(df, frequence_afternoon, on='user', how='left')
    del frequence_afternoon

    frequence_night = frequence_one_hour[(frequence_one_hour.hour <= 23) & (frequence_one_hour.hour >= 18)][
        ['user', 'days_diff', 'everyday_everyhour']]
    frequence_night['everyday_night'] = frequence_night['everyday_everyhour']
    frequence_night = frequence_night.groupby(['user', 'days_diff'])['everyday_night'].agg('sum').reset_index()
    frequence_night = frequence_night[['user', 'everyday_night']]
    frequence_night = frequence_night.groupby('user')['everyday_night'].agg({'mean', 'max', 'min', 'std'}).reset_index()
    frequence_night.columns = ['user',
                               'frequence_night_mean_{}'.format(prefix),
                               'frequence_night_max_{}'.format(prefix),
                               'frequence_night_min_{}'.format(prefix),
                               'frequence_night_std_{}'.format(prefix)]
    df = pd.merge(df, frequence_night, on='user', how='left')
    del frequence_night

    # 每一个用户这一天距离前面一段时间操作数的最大值，最小值，均值，标准差
    frequence_one_day_gap = other[['user', 'days_diff']]
    frequence_one_day_gap['everyday'] = 1
    frequence_one_day_gap = frequence_one_day_gap.groupby(['user', 'days_diff']).agg('count').reset_index()
    frequence_one_day_gap['everyday_before'] = frequence_one_day_gap.groupby('user')['everyday'].shift(1)
    frequence_one_day_gap['everyday_before_gap'] = frequence_one_day_gap['everyday'] - frequence_one_day_gap[
        'everyday_before']
    frequence_one_day_gap = frequence_one_day_gap[['user', 'everyday_before_gap']].groupby('user')[
        'everyday_before_gap'].agg({'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    frequence_one_day_gap.columns = ['user',
                                     'frequence_one_day_gap_sum_{}'.format(prefix),
                                     'frequence_one_day_gap_mean_{}'.format(prefix),
                                     'frequence_one_day_gap_max_{}'.format(prefix),
                                     'frequence_one_day_gap_min_{}'.format(prefix),
                                     'frequence_one_day_gap_std_{}'.format(prefix)]
    df = pd.merge(df, frequence_one_day_gap, on='user', how='left')
    del frequence_one_day_gap

    # 每一个用户这一小时距离前面一段时间操作数的最大值，最小值，均值，标准差
    frequence_one_hour_gap = other[['user', 'days_diff', 'hour']]
    frequence_one_hour_gap['everyhour'] = 1
    frequence_one_hour_gap = frequence_one_hour_gap.groupby(['user', 'days_diff', 'hour']).agg('count').reset_index()
    frequence_one_hour_gap['everyhour_before'] = frequence_one_hour_gap.groupby('user')['everyhour'].shift(1)
    frequence_one_hour_gap['everyhour_before_gap'] = frequence_one_hour_gap['everyhour'] - frequence_one_hour_gap[
        'everyhour_before']
    frequence_one_hour_gap = frequence_one_hour_gap[['user', 'everyhour_before_gap']].groupby('user')[
        'everyhour_before_gap'].agg({'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    frequence_one_hour_gap.columns = ['user',
                                      'frequence_one_hour_gap_sum_{}'.format(prefix),
                                      'frequence_one_hour_gap_mean_{}'.format(prefix),
                                      'frequence_one_hour_gap_max_{}'.format(prefix),
                                      'frequence_one_hour_gap_min_{}'.format(prefix),
                                      'frequence_one_hour_gap_std_{}'.format(prefix)]
    df = pd.merge(df, frequence_one_hour_gap, on='user', how='left')
    del frequence_one_hour_gap

    # 每一个用户这一周距离前面一段时间操作数的最大值，最小值，均值，标准差
    frequence_one_week_gap = other[['user', 'week']]
    frequence_one_week_gap['everyweek'] = 1
    frequence_one_week_gap = frequence_one_week_gap.groupby(['user', 'week']).agg('count').reset_index()
    frequence_one_week_gap['everyweek_before'] = frequence_one_week_gap.groupby('user')['everyweek'].shift(1)
    frequence_one_week_gap['everyweek_before_gap'] = frequence_one_week_gap['everyweek'] - frequence_one_week_gap[
        'everyweek_before']
    frequence_one_week_gap = frequence_one_week_gap[['user', 'everyweek_before_gap']].groupby('user')[
        'everyweek_before_gap'].agg({'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    frequence_one_week_gap.columns = ['user',
                                      'frequence_one_week_gap_sum_{}'.format(prefix),
                                      'frequence_one_week_gap_mean_{}'.format(prefix),
                                      'frequence_one_week_gap_max_{}'.format(prefix),
                                      'frequence_one_week_gap_min_{}'.format(prefix),
                                      'frequence_one_week_gap_std_{}'.format(prefix)]
    df = pd.merge(df, frequence_one_week_gap, on='user', how='left')
    del frequence_one_week_gap

    return df

def gen_user_amount_frequence(df,trans):
    """
    用户级：用户交易金额时间频率
    """
    # 用户每天的每小时交易金额的和、均值、最大值、最小值、标准差
    money_frequence_one_hour = trans[['user', 'days_diff', 'hour', 'amount']]
    money_frequence_one_hour = money_frequence_one_hour.groupby(['user', 'days_diff', 'hour'])['amount'].agg(
        'sum').reset_index()
    money_frequence_one_hour = money_frequence_one_hour[['user', 'amount']]
    money_frequence_one_hour['money_frequence_one_hour'] = money_frequence_one_hour['amount']
    money_frequence_one_hour = money_frequence_one_hour.groupby('user')['money_frequence_one_hour'].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    money_frequence_one_hour.columns = ['user',
                                        'money_frequence_one_hour_max',
                                        'money_frequence_one_hour_std',
                                        'money_frequence_one_hour_sum',
                                        'money_frequence_one_hour_min',
                                        'money_frequence_one_hour_mean']

    df = pd.merge(df, money_frequence_one_hour, on='user', how='left')
    del money_frequence_one_hour

    # 用户每天的每小时交易金额的上下两段时间的增量
    money_frequence_one_hour_gap = trans[['user', 'days_diff', 'hour', 'amount']]
    money_frequence_one_hour_gap = money_frequence_one_hour_gap.groupby(['user', 'days_diff', 'hour'])['amount'].agg(
        'sum').reset_index()
    money_frequence_one_hour_gap['before_hour_amt'] = money_frequence_one_hour_gap.groupby('user')['amount'].shift(1)
    money_frequence_one_hour_gap['money_frequence_one_hour_gap'] = money_frequence_one_hour_gap['amount'] - \
                                                                   money_frequence_one_hour_gap['before_hour_amt']
    money_frequence_one_hour_gap = money_frequence_one_hour_gap[['user', 'money_frequence_one_hour_gap']]
    money_frequence_one_hour_gap = money_frequence_one_hour_gap.groupby('user')['money_frequence_one_hour_gap'].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    money_frequence_one_hour_gap.columns = ['user',
                                            'money_frequence_one_hour_gap_sum',
                                            'money_frequence_one_hour_gap_max',
                                            'money_frequence_one_hour_gap_std',
                                            'money_frequence_one_hour_gap_min',
                                            'money_frequence_one_hour_gap_mean']
    df = pd.merge(df, money_frequence_one_hour_gap, on='user', how='left')
    del money_frequence_one_hour_gap



    # 用户每天的五个小时的和、均值、最大值、最小值、标准差(半夜、早上、下午、晚上)，gap、rate
    # for morning
    money_frequence_morning = trans[trans.hour <= 5][['user', 'days_diff', 'amount']]
    money_frequence_morning = money_frequence_morning.groupby(['user', 'days_diff'])['amount'].agg('sum').reset_index()
    money_frequence_morning = money_frequence_morning[['user', 'amount']]
    money_frequence_morning['trans_amt_sum'] = money_frequence_morning['amount']
    money_frequence_morning = money_frequence_morning.groupby('user')['trans_amt_sum'].agg(
        {'sum', 'mean', 'max', 'min', 'mean', 'std'}).reset_index()
    money_frequence_morning.columns = ['user',
                                       'money_frequence_morning_max',
                                       'money_frequence_morning_std',
                                       'money_frequence_morning_sum',
                                       'money_frequence_morning_min',
                                       'money_frequence_morning_mean']
    df = pd.merge(df, money_frequence_morning, on='user', how='left')
    del money_frequence_morning

    money_frequence_morning_gap = trans[trans.hour <= 5][['user', 'days_diff', 'amount']]
    money_frequence_morning_gap = money_frequence_morning_gap.groupby(['user', 'days_diff'])['amount'].agg(
        'sum').reset_index()
    money_frequence_morning_gap = money_frequence_morning_gap[['user', 'amount']]
    money_frequence_morning_gap['trans_amt_before'] = money_frequence_morning_gap.groupby('user')['amount'].shift(1)
    money_frequence_morning_gap['trans_amt_gap'] = money_frequence_morning_gap['amount'] - \
                                                   money_frequence_morning_gap['trans_amt_before']
    money_frequence_morning_gap = money_frequence_morning_gap[['user', 'trans_amt_gap']]
    money_frequence_morning_gap = money_frequence_morning_gap.groupby('user')['trans_amt_gap'].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    money_frequence_morning_gap.columns = ['user',
                                           'money_frequence_morning_gap_sum',
                                           'money_frequence_morning_gap_max',
                                           'money_frequence_morning_gap_std',
                                           'money_frequence_morning_gap_min',
                                           'money_frequence_morning_gap_mean']
    df = pd.merge(df, money_frequence_morning_gap, on='user', how='left')
    del money_frequence_morning_gap

    # for work_time
    money_frequence_work_time = trans[(trans.hour >= 6) & (trans.hour <= 11)][
        ['user', 'days_diff', 'amount']]
    money_frequence_work_time = money_frequence_work_time.groupby(['user', 'days_diff'])['amount'].agg('sum').reset_index()
    money_frequence_work_time = money_frequence_work_time[['user', 'amount']]
    money_frequence_work_time['trans_amt_sum'] = money_frequence_work_time['amount']
    money_frequence_work_time = money_frequence_work_time.groupby('user')['trans_amt_sum'].agg(
        {'sum', 'mean', 'max', 'min', 'mean', 'std'}).reset_index()
    money_frequence_work_time.columns = ['user',
                                         'money_frequence_work_time_max',
                                         'money_frequence_work_time_std',
                                         'money_frequence_work_time_sum',
                                         'money_frequence_work_time_min',
                                         'money_frequence_work_time_mean']
    df = pd.merge(df, money_frequence_work_time, on='user', how='left')
    del money_frequence_work_time

    money_frequence_work_time_gap = trans[(trans.hour >= 6) & (trans.hour <= 11)][
        ['user', 'days_diff', 'amount']]
    money_frequence_work_time_gap = money_frequence_work_time_gap.groupby(['user', 'days_diff'])['amount'].agg(
        'sum').reset_index()
    money_frequence_work_time_gap = money_frequence_work_time_gap[['user', 'amount']]
    money_frequence_work_time_gap['trans_amt_before'] = money_frequence_work_time_gap.groupby('user')['amount'].shift(
        1)
    money_frequence_work_time_gap['trans_amt_gap'] = money_frequence_work_time_gap['amount'] - \
                                                     money_frequence_work_time_gap['trans_amt_before']
    money_frequence_work_time_gap = money_frequence_work_time_gap[['user', 'trans_amt_gap']]
    money_frequence_work_time_gap = money_frequence_work_time_gap.groupby('user')['trans_amt_gap'].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    money_frequence_work_time_gap.columns = ['user',
                                             'money_frequence_work_time_gap_sum',
                                             'money_frequence_work_time_gap_max',
                                             'money_frequence_work_time_gap_std',
                                             'money_frequence_work_time_gap_min',
                                             'money_frequence_work_time_gap_mean']
    df = pd.merge(df, money_frequence_work_time_gap, on='user', how='left')
    del money_frequence_work_time_gap


    # for afternoon
    money_frequence_afternoon = trans[(trans.hour >= 12) & (trans.hour <= 17)][
        ['user', 'days_diff', 'amount']]
    money_frequence_afternoon = money_frequence_afternoon.groupby(['user', 'days_diff'])['amount'].agg('sum').reset_index()
    money_frequence_afternoon = money_frequence_afternoon[['user', 'amount']]
    money_frequence_afternoon['trans_amt_sum'] = money_frequence_afternoon['amount']
    money_frequence_afternoon = money_frequence_afternoon.groupby('user')['trans_amt_sum'].agg(
        {'sum', 'mean', 'max', 'min', 'mean', 'std'}).reset_index()
    money_frequence_afternoon.columns = ['user',
                                         'money_frequence_afternoon_max',
                                         'money_frequence_afternoon_std',
                                         'money_frequence_afternoon_sum',
                                         'money_frequence_afternoon_min',
                                         'money_frequence_afternoon_mean']
    df = pd.merge(df, money_frequence_afternoon, on='user', how='left')

    money_frequence_afternoon_gap = trans[(trans.hour >= 12) & (trans.hour <= 17)][
        ['user', 'days_diff', 'amount']]
    money_frequence_afternoon_gap = money_frequence_afternoon_gap.groupby(['user', 'days_diff'])['amount'].agg(
        'sum').reset_index()
    money_frequence_afternoon_gap = money_frequence_afternoon_gap[['user', 'amount']]
    money_frequence_afternoon_gap['trans_amt_before'] = money_frequence_afternoon_gap.groupby('user')['amount'].shift(
        1)
    money_frequence_afternoon_gap['trans_amt_gap'] = money_frequence_afternoon_gap['amount'] - \
                                                     money_frequence_afternoon_gap['trans_amt_before']
    money_frequence_afternoon_gap = money_frequence_afternoon_gap[['user', 'trans_amt_gap']]
    money_frequence_afternoon_gap = money_frequence_afternoon_gap.groupby('user')['trans_amt_gap'].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    money_frequence_afternoon_gap.columns = ['user',
                                             'money_frequence_afternoon_gap_sum',
                                             'money_frequence_afternoon_gap_max',
                                             'money_frequence_afternoon_gap_std',
                                             'money_frequence_afternoon_gap_min',
                                             'money_frequence_afternoon_gap_mean']
    df = pd.merge(df, money_frequence_afternoon_gap, on='user', how='left')
    del money_frequence_afternoon_gap

    # for night
    money_frequence_night = trans[(trans.hour >= 18) & (trans.hour <= 23)][
        ['user', 'days_diff', 'amount']]
    money_frequence_night = money_frequence_night.groupby(['user', 'days_diff'])['amount'].agg('sum').reset_index()
    money_frequence_night = money_frequence_night[['user', 'amount']]
    money_frequence_night['trans_amt_sum'] = money_frequence_night['amount']
    money_frequence_night = money_frequence_night.groupby('user')['trans_amt_sum'].agg(
        {'sum', 'mean', 'max', 'min', 'mean', 'std'}).reset_index()
    money_frequence_night.columns = ['user',
                                     'money_frequence_night_max',
                                     'money_frequence_night_std',
                                     'money_frequence_night_sum',
                                     'money_frequence_night_min',
                                     'money_frequence_night_mean']
    df = pd.merge(df, money_frequence_night, on='user', how='left')
    del money_frequence_night

    money_frequence_night_gap = trans[(trans.hour >= 18) & (trans.hour <= 23)][
        ['user', 'days_diff', 'amount']]
    money_frequence_night_gap = money_frequence_night_gap.groupby(['user', 'days_diff'])['amount'].agg(
        'sum').reset_index()
    money_frequence_night_gap = money_frequence_night_gap[['user', 'amount']]
    money_frequence_night_gap['trans_amt_before'] = money_frequence_night_gap.groupby('user')['amount'].shift(1)
    money_frequence_night_gap['trans_amt_gap'] = money_frequence_night_gap['amount'] - money_frequence_night_gap[
        'trans_amt_before']
    money_frequence_night_gap = money_frequence_night_gap[['user', 'trans_amt_gap']]
    money_frequence_night_gap = money_frequence_night_gap.groupby('user')['trans_amt_gap'].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    money_frequence_night_gap.columns = ['user',
                                         'money_frequence_night_gap_sum',
                                         'money_frequence_night_gap_max',
                                         'money_frequence_night_gap_std',
                                         'money_frequence_night_gap_min',
                                         'money_frequence_night_gap_mean']
    df = pd.merge(df, money_frequence_night_gap, on='user', how='left')
    del money_frequence_night_gap


    # 用户每天的交易金额和、最大值、最小值、均值、极差、gap、rate
    money_frequence_day = trans[['user', 'days_diff', 'amount']]
    money_frequence_day = money_frequence_day.groupby(['user', 'days_diff'])['amount'].agg('sum').reset_index()
    money_frequence_day = money_frequence_day[['user', 'amount']]
    money_frequence_day['trans_amt_day'] = money_frequence_day['amount']
    money_frequence_day = money_frequence_day.groupby('user')['trans_amt_day'].agg(
        {'max', 'min', 'mean', 'std'}).reset_index()
    money_frequence_day.columns = ['user',
                                   'money_frequence_day_max',
                                   'money_frequence_day_std',
                                   'money_frequence_day_min',
                                   'money_frequence_day_mean']
    df = pd.merge(df, money_frequence_day, on='user', how='left')
    del money_frequence_day

    money_frequence_day_gap = trans[['user', 'days_diff', 'amount']]
    money_frequence_day_gap = money_frequence_day_gap.groupby(['user', 'days_diff'])['amount'].agg('sum').reset_index()
    money_frequence_day_gap = money_frequence_day_gap[['user', 'amount']]
    money_frequence_day_gap['trans_amt_before'] = money_frequence_day_gap.groupby('user')['amount'].shift(1)
    money_frequence_day_gap['trans_amt_gap'] = money_frequence_day_gap['amount'] - money_frequence_day_gap[
        'trans_amt_before']
    money_frequence_day_gap = money_frequence_day_gap[['user', 'trans_amt_gap']]
    money_frequence_day_gap = money_frequence_day_gap.groupby('user')['trans_amt_gap'].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    money_frequence_day_gap.columns = ['user',
                                       'money_frequence_day_gap_sum',
                                       'money_frequence_day_gap_max',
                                       'money_frequence_day_gap_std',
                                       'money_frequence_day_gap_min',
                                       'money_frequence_day_gap_mean']
    df = pd.merge(df, money_frequence_day_gap, on='user', how='left')
    del money_frequence_day_gap

    return df

def gen_user_nunique_features(df, value, prefix):
    group_df = df.groupby(['user'])[value].agg({
        'user_{}_{}_nuniq'.format(prefix, value): 'nunique'
    }).reset_index()
    return group_df

def gen_user_count_features(df, value, prefix):
    group_df = df.groupby(['user'])[value].agg({
        'user_{}_{}_count'.format(prefix, value): 'count'
    }).reset_index()
    return group_df

def gen_user_sum_features(df,prefix):
    df['count'] = 1
    group_df = df.groupby('user')['count'].agg({
        'user_{}_sum'.format(prefix): 'sum'
    }).reset_index()
    del df['count']

    return group_df

def gen_user_null_features(df, value, prefix):
    df['is_null'] = 0
    df.loc[df[value].isnull(), 'is_null'] = 1

    group_df = df.groupby(['user'])['is_null'].agg({'user_{}_{}_null_cnt'.format(prefix, value): 'sum',
                                                    'user_{}_{}_null_ratio'.format(prefix, value): 'mean'}).reset_index()
    return group_df

def gen_user_tfidf_features(df, value,prefix,n_com):
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    group_df.columns = ['user', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_vec = TfidfVectorizer()
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=n_com, n_iter=20, random_state=2020)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_tfidf_{}_{}_{}'.format(value, prefix, i) for i in range(n_com)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df

def gen_user_countvec_features(df, value,prefix,n_com):
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    group_df.columns = ['user', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_vec = CountVectorizer()
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=n_com, n_iter=20, random_state=2020)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_countvec_{}_{}_{}'.format(value, prefix, i) for i in range(n_com)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df

def kfold_stats_feature(train, test, feats, k,random_):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = ['label']
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_trn = train.iloc[trn_idx]
                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = train[f].mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            test[colname] = None
            order_label = train.groupby([feat])[f].mean()
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = train[f].mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test

def get_user_pay(data,transaction_data):
    """
    时间差特征
    """
    # 每一天用户操作的时间差
    every_days_diff_time2_gap = transaction_data[['user', 'days_diff']]
    every_days_diff_time2_gap = every_days_diff_time2_gap.sort_values(['user', 'days_diff'])
    every_days_diff_time2_gap['last_days_diff'] = every_days_diff_time2_gap.groupby('user')['days_diff'].shift(1)
    every_days_diff_time2_gap['last_days_diff_gap'] = every_days_diff_time2_gap['days_diff'] - every_days_diff_time2_gap['last_days_diff']
    every_days_diff_time2_gap = every_days_diff_time2_gap[['user', 'last_days_diff_gap']]
    every_days_diff_time2_gap = every_days_diff_time2_gap.groupby('user')['last_days_diff_gap'].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    every_days_diff_time2_gap.columns = ['user', 'every_days_diff_time2_gap_sum', 'every_days_diff_time2_gap_std',
                                   'every_days_diff_time2_gap_min', 'every_days_diff_time2_gap_mean', 'every_days_diff_time2_gap_max']
    data = pd.merge(data, every_days_diff_time2_gap, on='user', how='left')
    del every_days_diff_time2_gap

    # 每三天用户操作的时间差
    every_three_days_diff_time2_gap = transaction_data[['user', 'days_diff']]
    every_three_days_diff_time2_gap = every_three_days_diff_time2_gap.sort_values(['user', 'days_diff'])
    every_three_days_diff_time2_gap['last_three_days_diff'] = every_three_days_diff_time2_gap.groupby('user')['days_diff'].shift(3)
    every_three_days_diff_time2_gap['last_three_days_diff_gap'] = every_three_days_diff_time2_gap['days_diff'] - every_three_days_diff_time2_gap[
        'last_three_days_diff']
    every_three_days_diff_time2_gap = every_three_days_diff_time2_gap[['user', 'last_three_days_diff_gap']]
    every_three_days_diff_time2_gap = every_three_days_diff_time2_gap.groupby('user')['last_three_days_diff_gap'].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    every_three_days_diff_time2_gap.columns = ['user', 'every_three_days_diff_time2_gap_sum', 'every_three_days_diff_time2_gap_std',
                                         'every_three_days_diff_time2_gap_min', 'every_three_days_diff_time2_gap_mean',
                                         'every_three_days_diff_time2_gap_max']
    data = pd.merge(data, every_three_days_diff_time2_gap, on='user', how='left')
    del every_three_days_diff_time2_gap

    # 每七天用户操作的时间差
    every_sixteen_days_diff_time2_gap = transaction_data[['user', 'days_diff']]
    every_sixteen_days_diff_time2_gap = every_sixteen_days_diff_time2_gap.sort_values(['user', 'days_diff'])
    every_sixteen_days_diff_time2_gap['last_sixteen_days_diff'] = every_sixteen_days_diff_time2_gap.groupby('user')['days_diff'].shift(7)
    every_sixteen_days_diff_time2_gap['last_sixteen_days_diff_gap'] = every_sixteen_days_diff_time2_gap['days_diff'] - \
                                                          every_sixteen_days_diff_time2_gap['last_sixteen_days_diff']
    every_sixteen_days_diff_time2_gap = every_sixteen_days_diff_time2_gap[['user', 'last_sixteen_days_diff_gap']]
    every_sixteen_days_diff_time2_gap = every_sixteen_days_diff_time2_gap.groupby('user')['last_sixteen_days_diff_gap'].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    every_sixteen_days_diff_time2_gap.columns = ['user', 'every_sixteen_days_diff_time2_gap_sum', 'every_sixteen_days_diff_time2_gap_std',
                                           'every_sixteen_days_diff_time2_gap_min', 'every_sixteen_days_diff_time2_gap_mean',
                                           'every_sixteen_days_diff_time2_gap_max']
    data = pd.merge(data, every_sixteen_days_diff_time2_gap, on='user', how='left')
    del every_sixteen_days_diff_time2_gap

    return data

def amt_label_frequence(data, temp, x):
    # 每天的每一小时用该x(如ip)进行交易的钱数
    every_days_diff_hour_amt = temp[[x, 'days_diff', 'hour', 'amount']]
    every_days_diff_hour_amt['every_days_diff_' + x + '_amt'] = every_days_diff_hour_amt['amount']
    every_days_diff_hour_amt = every_days_diff_hour_amt.groupby([x, 'days_diff', 'hour'])['every_days_diff_' + x + '_amt'].agg(
        'sum').reset_index()
    every_days_diff_hour_amt = pd.merge(temp[['user', x, 'days_diff', 'hour']], every_days_diff_hour_amt, on=[x, 'days_diff', 'hour'],
                                  how='left')
    every_days_diff_hour_amt = every_days_diff_hour_amt[['user', 'every_days_diff_' + x + '_amt']]
    every_days_diff_hour_amt = every_days_diff_hour_amt.groupby('user')['every_days_diff_' + x + '_amt'].agg(
        {'mean', 'max', 'min', 'std'}).reset_index()
    every_days_diff_hour_amt.columns = ['user', 'every_days_diff_hour_amt_std_' + x, 'every_days_diff_hour_amt_mean_' + x,
                                  'every_days_diff_hour_amt_min_' + x, 'every_days_diff_hour_amt_max_' + x]
    data = pd.merge(data, every_days_diff_hour_amt, on='user', how='left')
    del every_days_diff_hour_amt

    # 每天的每一小时用该x(如ip)进行交易的钱数的gap
    every_days_diff_hour_amt_gap = temp[[x, 'days_diff', 'hour', 'amount']]
    every_days_diff_hour_amt_gap['every_days_diff_' + x + '_amt'] = every_days_diff_hour_amt_gap['amount']
    every_days_diff_hour_amt_gap = every_days_diff_hour_amt_gap.groupby([x, 'days_diff', 'hour'])['every_days_diff_' + x + '_amt'].agg(
        'sum').reset_index()
    every_days_diff_hour_amt_gap = pd.merge(temp[['user', x, 'days_diff', 'hour']], every_days_diff_hour_amt_gap, on=[x, 'days_diff', 'hour'],
                                      how='left')
    every_days_diff_hour_amt_gap = every_days_diff_hour_amt_gap[['user', 'every_days_diff_' + x + '_amt']]
    every_days_diff_hour_amt_gap['before_every_days_diff_hour_amt'] = every_days_diff_hour_amt_gap.groupby('user')[
        'every_days_diff_' + x + '_amt'].shift(1)
    every_days_diff_hour_amt_gap['every_days_diff_hour_amt_gap_' + 'of_' + x] = every_days_diff_hour_amt_gap['every_days_diff_' + x + '_amt'] - \
                                                                    every_days_diff_hour_amt_gap['before_every_days_diff_hour_amt']
    every_days_diff_hour_amt_gap = every_days_diff_hour_amt_gap[['user', 'every_days_diff_hour_amt_gap_' + 'of_' + x]]
    every_days_diff_hour_amt_gap = every_days_diff_hour_amt_gap.groupby('user')['every_days_diff_hour_amt_gap_' + 'of_' + x].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    every_days_diff_hour_amt_gap.columns = ['user', 'every_days_diff_hour_amt_gap_sum_' + x, 'every_days_diff_hour_amt_gap_std_' + x,
                                      'every_days_diff_hour_amt_gap_mean_' + x, 'every_days_diff_hour_amt_gap_min_' + x,
                                      'every_days_diff_hour_amt_gap_max_' + x]
    data = pd.merge(data, every_days_diff_hour_amt_gap, on='user', how='left')
    del every_days_diff_hour_amt_gap

    # 每天用该x(如ip)进行交易的钱数
    every_days_diff_amt = temp[[x, 'days_diff', 'amount']]
    every_days_diff_amt['every_days_diff_' + x + '_amt'] = every_days_diff_amt['amount']
    every_days_diff_amt = every_days_diff_amt.groupby([x, 'days_diff'])['every_days_diff_' + x + '_amt'].agg('sum').reset_index()
    every_days_diff_amt = pd.merge(temp[['user', x, 'days_diff']], every_days_diff_amt, on=[x, 'days_diff'], how='left')
    every_days_diff_amt = every_days_diff_amt[['user', 'every_days_diff_' + x + '_amt']]
    every_days_diff_amt = every_days_diff_amt.groupby('user')['every_days_diff_' + x + '_amt'].agg(
        {'mean', 'max', 'min', 'std'}).reset_index()
    every_days_diff_amt.columns = ['user', 'every_days_diff_amt_std_' + x, 'every_days_diff_amt_mean_' + x, 'every_days_diff_amt_min_' + x,
                             'every_days_diff_amt_max_' + x]
    data = pd.merge(data, every_days_diff_amt, on='user', how='left')
    del every_days_diff_amt

    # 每天用该x(如ip)进行交易的钱数的gap
    every_days_diff_amt_gap = temp[[x, 'days_diff', 'amount']]
    every_days_diff_amt_gap['every_days_diff_' + x + '_amt'] = every_days_diff_amt_gap['amount']
    every_days_diff_amt_gap = every_days_diff_amt_gap.groupby([x, 'days_diff'])['every_days_diff_' + x + '_amt'].agg('sum').reset_index()
    every_days_diff_amt_gap = pd.merge(temp[['user', x, 'days_diff']], every_days_diff_amt_gap, on=[x, 'days_diff'], how='left')
    every_days_diff_amt_gap = every_days_diff_amt_gap[['user', 'every_days_diff_' + x + '_amt']]
    every_days_diff_amt_gap['before_every_days_diff_amt'] = every_days_diff_amt_gap.groupby('user')['every_days_diff_' + x + '_amt'].shift(1)
    every_days_diff_amt_gap['every_days_diff_amt_gap_' + 'of_' + x] = every_days_diff_amt_gap['every_days_diff_' + x + '_amt'] - \
                                                          every_days_diff_amt_gap['before_every_days_diff_amt']
    every_days_diff_amt_gap = every_days_diff_amt_gap[['user', 'every_days_diff_amt_gap_' + 'of_' + x]]
    every_days_diff_amt_gap = every_days_diff_amt_gap.groupby('user')['every_days_diff_amt_gap_' + 'of_' + x].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    every_days_diff_amt_gap.columns = ['user', 'every_days_diff_amt_gap_sum_' + x, 'every_days_diff_amt_gap_std_' + x,
                                 'every_days_diff_amt_gap_mean_' + x, 'every_days_diff_amt_gap_min_' + x,
                                 'every_days_diff_amt_gap_max_' + x]
    data = pd.merge(data, every_days_diff_amt_gap, on='user', how='left')
    del every_days_diff_amt_gap

    # 每天半夜用该x(如ip)进行交易的钱数
    every_morning_amt = temp[temp.hour <= 5][[x, 'days_diff', 'amount']]
    every_morning_amt['every_morning_' + x + '_amt'] = every_morning_amt['amount']
    every_morning_amt = every_morning_amt.groupby([x, 'days_diff'])['every_morning_' + x + '_amt'].agg('sum').reset_index()
    every_morning_amt = pd.merge(temp[['user', x, 'days_diff']], every_morning_amt, on=[x, 'days_diff'], how='left')
    every_morning_amt = every_morning_amt[['user', 'every_morning_' + x + '_amt']]
    every_morning_amt = every_morning_amt.groupby('user')['every_morning_' + x + '_amt'].agg(
        {'mean', 'max', 'min', 'std'}).reset_index()
    every_morning_amt.columns = ['user', 'every_morning_amt_std_' + x, 'every_morning_amt_mean_' + x,
                                 'every_morning_amt_min_' + x, 'every_morning_amt_max_' + x]
    data = pd.merge(data, every_morning_amt, on='user', how='left')
    del every_morning_amt

    # 每天半夜用该x(如ip)进行交易的钱数的gap
    every_morning_amt_gap = temp[temp.hour <= 5][[x, 'days_diff', 'amount']]
    every_morning_amt_gap['every_morning_' + x + '_amt'] = every_morning_amt_gap['amount']
    every_morning_amt_gap = every_morning_amt_gap.groupby([x, 'days_diff'])['every_morning_' + x + '_amt'].agg(
        'sum').reset_index()
    every_morning_amt_gap = pd.merge(temp[['user', x, 'days_diff']], every_morning_amt_gap, on=[x, 'days_diff'], how='left')
    every_morning_amt_gap = every_morning_amt_gap[['user', 'every_morning_' + x + '_amt']]
    every_morning_amt_gap['before_every_morning_amt'] = every_morning_amt_gap.groupby('user')[
        'every_morning_' + x + '_amt'].shift(1)
    every_morning_amt_gap['every_morning_amt_gap_' + 'of_' + x] = every_morning_amt_gap['every_morning_' + x + '_amt'] - \
                                                                  every_morning_amt_gap['before_every_morning_amt']
    every_morning_amt_gap = every_morning_amt_gap[['user', 'every_morning_amt_gap_' + 'of_' + x]]
    every_morning_amt_gap = every_morning_amt_gap.groupby('user')['every_morning_amt_gap_' + 'of_' + x].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    every_morning_amt_gap.columns = ['user', 'every_morning_amt_gap_sum_' + x, 'every_morning_amt_gap_std_' + x,
                                     'every_morning_amt_gap_mean_' + x, 'every_morning_amt_gap_min_' + x,
                                     'every_morning_amt_gap_max_' + x]
    data = pd.merge(data, every_morning_amt_gap, on='user', how='left')
    del every_morning_amt_gap

    # 每天早上用该x(如ip)进行交易的钱数
    every_work_time_amt = temp[(temp.hour <= 11) & (temp.hour >= 6)][[x, 'days_diff', 'amount']]
    every_work_time_amt['every_work_time_' + x + '_amt'] = every_work_time_amt['amount']
    every_work_time_amt = every_work_time_amt.groupby([x, 'days_diff'])['every_work_time_' + x + '_amt'].agg(
        'sum').reset_index()
    every_work_time_amt = pd.merge(temp[['user', x, 'days_diff']], every_work_time_amt, on=[x, 'days_diff'], how='left')
    every_work_time_amt = every_work_time_amt[['user', 'every_work_time_' + x + '_amt']]
    every_work_time_amt = every_work_time_amt.groupby('user')['every_work_time_' + x + '_amt'].agg(
        {'mean', 'max', 'min', 'std'}).reset_index()
    every_work_time_amt.columns = ['user', 'every_work_time_amt_std_' + x, 'every_work_time_amt_mean_' + x,
                                   'every_work_time_amt_min_' + x, 'every_work_time_amt_max_' + x]
    data = pd.merge(data, every_work_time_amt, on='user', how='left')
    del every_work_time_amt

    # 每天早上用该x(如ip)进行交易的钱数的gap
    every_work_time_amt_gap = temp[(temp.hour <= 11) & (temp.hour >= 6)][[x, 'days_diff', 'amount']]
    every_work_time_amt_gap['every_work_time_' + x + '_amt'] = every_work_time_amt_gap['amount']
    every_work_time_amt_gap = every_work_time_amt_gap.groupby([x, 'days_diff'])['every_work_time_' + x + '_amt'].agg(
        'sum').reset_index()
    every_work_time_amt_gap = pd.merge(temp[['user', x, 'days_diff']], every_work_time_amt_gap, on=[x, 'days_diff'], how='left')
    every_work_time_amt_gap = every_work_time_amt_gap[['user', 'every_work_time_' + x + '_amt']]
    every_work_time_amt_gap['before_every_work_time_amt'] = every_work_time_amt_gap.groupby('user')[
        'every_work_time_' + x + '_amt'].shift(1)
    every_work_time_amt_gap['every_work_time_amt_gap_' + 'of_' + x] = every_work_time_amt_gap[
                                                                          'every_work_time_' + x + '_amt'] - \
                                                                      every_work_time_amt_gap[
                                                                          'before_every_work_time_amt']
    every_work_time_amt_gap = every_work_time_amt_gap[['user', 'every_work_time_amt_gap_' + 'of_' + x]]
    every_work_time_amt_gap = every_work_time_amt_gap.groupby('user')['every_work_time_amt_gap_' + 'of_' + x].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    every_work_time_amt_gap.columns = ['user', 'every_work_time_amt_gap_sum_' + x, 'every_work_time_amt_gap_std_' + x,
                                       'every_work_time_amt_gap_mean_' + x, 'every_work_time_amt_gap_min_' + x,
                                       'every_work_time_amt_gap_max_' + x]
    data = pd.merge(data, every_work_time_amt_gap, on='user', how='left')
    del every_work_time_amt_gap

    # 每天下午用该x(如ip)进行交易的钱数
    every_afternoon_amt = temp[(temp.hour <= 17) & (temp.hour >= 12)][[x, 'days_diff', 'amount']]
    every_afternoon_amt['every_afternoon_' + x + '_amt'] = every_afternoon_amt['amount']
    every_afternoon_amt = every_afternoon_amt.groupby([x, 'days_diff'])['every_afternoon_' + x + '_amt'].agg(
        'sum').reset_index()
    every_afternoon_amt = pd.merge(temp[['user', x, 'days_diff']], every_afternoon_amt, on=[x, 'days_diff'], how='left')
    every_afternoon_amt = every_afternoon_amt[['user', 'every_afternoon_' + x + '_amt']]
    every_afternoon_amt = every_afternoon_amt.groupby('user')['every_afternoon_' + x + '_amt'].agg(
        {'mean', 'max', 'min', 'std'}).reset_index()
    every_afternoon_amt.columns = ['user', 'every_afternoon_amt_std_' + x, 'every_afternoon_amt_mean_' + x,
                                   'every_afternoon_amt_min_' + x, 'every_afternoon_amt_max_' + x]
    data = pd.merge(data, every_afternoon_amt, on='user', how='left')
    del every_afternoon_amt

    # 每天下午用该x(如ip)进行交易的钱数的gap
    every_afternoon_amt_gap = temp[(temp.hour <= 17) & (temp.hour >= 12)][[x, 'days_diff', 'amount']]
    every_afternoon_amt_gap['every_afternoon_' + x + '_amt'] = every_afternoon_amt_gap['amount']
    every_afternoon_amt_gap = every_afternoon_amt_gap.groupby([x, 'days_diff'])['every_afternoon_' + x + '_amt'].agg(
        'sum').reset_index()
    every_afternoon_amt_gap = pd.merge(temp[['user', x, 'days_diff']], every_afternoon_amt_gap, on=[x, 'days_diff'], how='left')
    every_afternoon_amt_gap = every_afternoon_amt_gap[['user', 'every_afternoon_' + x + '_amt']]
    every_afternoon_amt_gap['before_every_afternoon_amt'] = every_afternoon_amt_gap.groupby('user')[
        'every_afternoon_' + x + '_amt'].shift(1)
    every_afternoon_amt_gap['every_afternoon_amt_gap_' + 'of_' + x] = every_afternoon_amt_gap[
                                                                          'every_afternoon_' + x + '_amt'] - \
                                                                      every_afternoon_amt_gap[
                                                                          'before_every_afternoon_amt']
    every_afternoon_amt_gap = every_afternoon_amt_gap[['user', 'every_afternoon_amt_gap_' + 'of_' + x]]
    every_afternoon_amt_gap = every_afternoon_amt_gap.groupby('user')['every_afternoon_amt_gap_' + 'of_' + x].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    every_afternoon_amt_gap.columns = ['user', 'every_afternoon_amt_gap_sum_' + x, 'every_afternoon_amt_gap_std_' + x,
                                       'every_afternoon_amt_gap_mean_' + x, 'every_afternoon_amt_gap_min_' + x,
                                       'every_afternoon_amt_gap_max_' + x]
    data = pd.merge(data, every_afternoon_amt_gap, on='user', how='left')
    del every_afternoon_amt_gap

    # 每天晚上用该x(如ip)进行交易的钱数
    every_night_amt = temp[(temp.hour <= 23) & (temp.hour >= 18)][[x, 'days_diff', 'amount']]
    every_night_amt['every_night_' + x + '_amt'] = every_night_amt['amount']
    every_night_amt = every_night_amt.groupby([x, 'days_diff'])['every_night_' + x + '_amt'].agg('sum').reset_index()
    every_night_amt = pd.merge(temp[['user', x, 'days_diff']], every_night_amt, on=[x, 'days_diff'], how='left')
    every_night_amt = every_night_amt[['user', 'every_night_' + x + '_amt']]
    every_night_amt = every_night_amt.groupby('user')['every_night_' + x + '_amt'].agg(
        {'mean', 'max', 'min', 'std'}).reset_index()
    every_night_amt.columns = ['user', 'every_night_amt_std_' + x, 'every_night_amt_mean_' + x,
                               'every_night_amt_min_' + x, 'every_night_amt_max_' + x]
    data = pd.merge(data, every_night_amt, on='user', how='left')
    del every_night_amt

    # 每天晚上用该x(如ip)进行交易的钱数的gap
    every_night_amt_gap = temp[(temp.hour <= 23) & (temp.hour >= 18)][[x, 'days_diff', 'amount']]
    every_night_amt_gap['every_night_' + x + '_amt'] = every_night_amt_gap['amount']
    every_night_amt_gap = every_night_amt_gap.groupby([x, 'days_diff'])['every_night_' + x + '_amt'].agg('sum').reset_index()
    every_night_amt_gap = pd.merge(temp[['user', x, 'days_diff']], every_night_amt_gap, on=[x, 'days_diff'], how='left')
    every_night_amt_gap = every_night_amt_gap[['user', 'every_night_' + x + '_amt']]
    every_night_amt_gap['before_every_night_amt'] = every_night_amt_gap.groupby('user')[
        'every_night_' + x + '_amt'].shift(1)
    every_night_amt_gap['every_night_amt_gap_' + 'of_' + x] = every_night_amt_gap['every_night_' + x + '_amt'] - \
                                                              every_night_amt_gap['before_every_night_amt']
    every_night_amt_gap = every_night_amt_gap[['user', 'every_night_amt_gap_' + 'of_' + x]]
    every_night_amt_gap = every_night_amt_gap.groupby('user')['every_night_amt_gap_' + 'of_' + x].agg(
        {'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    every_night_amt_gap.columns = ['user', 'every_night_amt_gap_sum_' + x, 'every_night_amt_gap_std_' + x,
                                   'every_night_amt_gap_mean_' + x, 'every_night_amt_gap_min_' + x,
                                   'every_night_amt_gap_max_' + x]
    data = pd.merge(data, every_night_amt_gap, on='user', how='left')
    del every_night_amt_gap

    print(x + ' is over!')
    return data

def get_two_count(data, t_data, x1, x2):
    """
    获得x1和x2的交叉count
    """
    temp = t_data[[x1, x2, 'user']]

    operation_or_transaction_data = temp.copy()
    operation_or_transaction_data.drop_duplicates(inplace=True)

    temp[x1 + '_' + x2 + '_count'] = temp['user']
    temp = temp.groupby([x1, x2])[x1 + '_' + x2 + '_count'].agg('nunique').reset_index()

    operation_or_transaction_data = pd.merge(operation_or_transaction_data, temp, on=[x1, x2], how='left')
    operation_or_transaction_data[x1 + '_' + x2 + '_count_user'] = 1
    operation_or_transaction_data = operation_or_transaction_data.groupby('user')[x1 + '_' + x2 + '_count_user'].agg(
        'sum').reset_index()

    data = pd.merge(data, operation_or_transaction_data, on='user', how='left')
    del operation_or_transaction_data

    return data


def get_two_shuxing_count(data, t_data, x1, x2):
    """
    groupby(x1)[x2].agg('nunique').reset_index()
    """
    user = t_data[['user', x1]].drop_duplicates()
    temp = t_data[[x1, x2]]
    temp[x2 + '_of_nunique_' + x1] = temp[x2]
    temp = temp.groupby(x1)[x2 + '_of_nunique_' + x1].agg('nunique').reset_index()

    user = pd.merge(user, temp, on=x1, how='left')
    user = user.drop(x1, axis=1)
    user = user.groupby('user')[x2 + '_of_nunique_' + x1].agg('sum').reset_index()

    data = pd.merge(data, user, on='user', how='left')
    del user

    return data


def get_three_count(data, t_data, x1, x2, x3):
    temp = t_data[[x1, x2, x3, 'user']]

    operation_or_transaction_data = temp.copy()
    operation_or_transaction_data.drop_duplicates(inplace=True)

    temp[x1 + '_' + x2 + '_' + x3 + '_count'] = temp['user']
    temp = temp.groupby([x1, x2, x3])[x1 + '_' + x2 + '_' + x3 + '_count'].agg('nunique').reset_index()

    operation_or_transaction_data = pd.merge(operation_or_transaction_data, temp, on=[x1, x2, x3], how='left')
    operation_or_transaction_data[x1 + '_' + x2 + '_' + x3 + '_count_user'] = 1
    operation_or_transaction_data = operation_or_transaction_data.groupby('user')[
        x1 + '_' + x2 + '_' + x3 + '_count_user'].agg('sum').reset_index()

    data = pd.merge(data, operation_or_transaction_data, on='user', how='left')
    return data

def get_word2vec_feature(seq,emb,feat,ikx,prefix,ext='',feature=[]):
    sentence = [[str(x) for x in x] for x in seq]
    # if os.path.exists('w2v_model_{}_{}_{}.model'.format('_'.join(feat),ext,prefix)):
    #     model = Word2Vec.load('w2v_model_{}_{}_{}.model'.format('_'.join(feat),ext,prefix))
    # else:
    model = Word2Vec(sentence, size=emb, window=5, min_count=1, workers=10, iter=10, sg=1, seed=42)
        # model.save('w2v_model_{}_{}_{}.model'.format('_'.join(feat),ext,prefix))
    return model

def train_vec(res,f,dim,prefix):
    import pickle
    res[f].fillna(-1, inplace=True)
    model = get_word2vec_feature(res[f].values.tolist(), dim, ['user', f], f, prefix, ext=str(dim), feature=[])

    emb_matrix = []
    for col in tqdm(res[f].values):
        tmp = np.zeros(shape=(dim))
        for seq in col:
            tmp = tmp + model[str(seq)] / len(col)

        emb_matrix.append(tmp)
    emb_matrix = np.array(emb_matrix)

    for i in range(dim):
        res['{}_{}_{}'.format('user', f + '_emb', i)] = emb_matrix[:, i]
    del res[f]
    fp = open(prefix + '_' + f + "_emb.pkl", "wb+")
    pickle.dump(res, fp)

def gen_features(df, op, trans):
    df.drop(['service3_level'], axis=1, inplace=True)   #该特征null值占比90%以上在训练集和测试集
    # base
    df['product7_fail_ratio'] = df['product7_fail_cnt'] / df['product7_cnt']
    for col in tqdm(['age','using_time']):
        df[col + '_max'] = df[col].max()-df[col]
        df[col + '_min'] = df[col]-df[col].min()
        df[col + '_median'] = df[col]-df[col].median()
        df[col + '_mean'] = df[col]-df[col].mean()

    for col in tqdm(['province', 'city', 'regist_type']):
        df[col + '_user_count'] = df.groupby([col])['user'].transform('count')
        df[col + '_count'] = df[col].map(df[col].value_counts(normalize=True))

    df['product_amount_sum'] = df[
        ['product1_amount', 'product2_amount', 'product3_amount', 'product4_amount', 'product5_amount',
         'product6_amount']].sum(axis=1)
    df['product_amount_max'] = df[
        ['product1_amount', 'product2_amount', 'product3_amount', 'product4_amount', 'product5_amount',
         'product6_amount']].max(axis=1)
    df['card_sum_cnt'] = df[['card_a_cnt','card_b_cnt','card_c_cnt','card_d_cnt']].sum(axis=1)
    df['card_max_cnt'] = df[['card_a_cnt', 'card_b_cnt', 'card_c_cnt', 'card_d_cnt']].max(axis=1)
    df['card_div_cnt'] = df['card_a_cnt'] / df['card_sum_cnt']
    df['op_sum_cnt'] = df[['op1_cnt','op2_cnt']].sum(axis=1)
    df['op_div_cnt'] = df['op1_cnt']/df['op2_cnt']
    df['login_cnt_period12'] = df['login_cnt_period1'] + df['login_cnt_period2']
    df['login_cnt'] = df['login_days_cnt'] * df['login_cnt_avg']
    df['acc_card_ratio'] = df['acc_count'] / df['card_sum_cnt']
    df['time_age'] = df['using_time'] / df['age']
    df['card_time'] = df['card_sum_cnt'] / df['using_time']
    df['op_time'] = df['op_sum_cnt'] / df['using_time']
    df['service_sum_cnt'] = df['service1_cnt'] + df['service2_cnt']
    df['service_cnt_time'] = df['service_sum_cnt'] / df['using_time']
    df['service1_amtt_time'] = df['service1_amt'] / df['using_time']
    df['agreement_time'] = df['agreement_total'] / df['using_time']
    df['acc_time'] = df['acc_count'] / df['using_time']
    df['fail_time'] = df['product7_fail_cnt'] / df['using_time']
    df['cnt_time'] = df['product7_cnt'] / df['using_time']


    col1 = ['ip_cnt']
    col2 = ['login_cnt_period1', 'login_cnt_period2', 'login_cnt_avg', 'login_days_cnt','login_cnt_period12']
    for f1 in col1:
        for f2 in col2:
            df[f1 + '_' + f2] = df[f1] / df[f2]
    # trans
    df = df.merge(gen_user_amount_features(trans), on=['user'], how='left')  #用户和交易金额的关系
    df['amount_time'] = df['user_amount_sum'] / df['using_time']
    for col in tqdm(['platform',  'type1', 'type2', 'ip']):
        df = df.merge(gen_user_nunique_features(df=trans, value=col, prefix='trans'), on=['user'], how='left')   #生成用户的对应特征的数量
    df['user_amount_per_cnt'] = df['user_amount_sum'] / df['user_amount_cnt']
    df = df.merge(gen_user_group_amount_features(df=trans, value='platform'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(df=trans, value='type1'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(df=trans, value='type2'), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=27), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=23), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=15), on=['user'], how='left')
    df = df.merge(gen_user_null_features(df=trans, value='ip', prefix='trans'), on=['user'], how='left')
    df = df.merge(gen_user_null_features(df=trans, value='type2', prefix='trans'), on=['user'], how='left')

    group_df = trans[trans['type1']=='45a1168437c708ff'].groupby(['user'])['days_diff'].agg({'user_type1_45a1168437c708ff_min_day': 'min'}).reset_index()
    df = df.merge(group_df, on=['user'], how='left')
    df = df.merge(gen_user_sum_features(trans, 'trans'), on=['user'], how='left')
    for f in ['days_diff', 'week', 'hour']:
        df_temp = trans.groupby(['user'
                                    ])[f].agg({f + '_trmean': 'mean',
                                               f + '_trstd': 'std',
                                               f + '_trmax': 'max',
                                               f + '_trmin': 'min',
                                               f + '_trnu': 'nunique'
                                               }).reset_index()
        df_temp[f + '_trrange'] = df_temp[f + '_trmax'] - df_temp[f + '_trmin']
        df = df.merge(df_temp, how='left')
    df['user_amount_per_days'] = df['user_amount_sum'] / df['days_diff_trnu']

    # op
    for col in tqdm(['op_type', 'op_mode', 'op_device', 'ip', 'net_type', 'channel', 'ip_3']):
        df = df.merge(gen_user_nunique_features(df=op, value=col, prefix='op'), on=['user'],
                      how='left')  # 生成用户的对应特征的数量
    df = df.merge(gen_user_tfidf_features(df=op, value='op_mode',prefix='op',n_com=10), on=['user'], how='left')
    df = df.merge(gen_user_tfidf_features(df=op, value='op_type',prefix='op',n_com=10), on=['user'], how='left')
    df = df.merge(gen_user_tfidf_features(df=op, value='op_device', prefix='op', n_com=10), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_mode',prefix='op',n_com=10), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_type',prefix='op',n_com=10), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_device', prefix='op', n_com=10), on=['user'], how='left')
    df = df.merge(gen_user_sum_features(op,'op'),on=['user'],how='left')
    df = df.merge(gen_user_null_features(df=op, value='net_type', prefix='op'), on=['user'], how='left')
    df['operation_transaction_count_gap'] = df['user_op_sum'] - df['user_trans_sum']

    for f in ['days_diff', 'week', 'hour']:
        df_temp = op.groupby(['user'
                                 ])[f].agg({f + '_opmean': 'mean',
                                            f + '_opstd': 'std',
                                            f + '_opmax': 'max',
                                            f + '_opmin': 'min',
                                            f + '_opnu': 'nunique'
                                            }).reset_index()
        df_temp[f + '_oprange'] = df_temp[f + '_opmax'] - df_temp[f + '_opmin']
        df = df.merge(df_temp, how='left')

    a = ['op_type', 'op_mode', 'op_device', 'ip','channel','net_type']
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            df = get_two_count(df, op, a[i], a[j])

    b = ['platform', 'tunnel_out', 'type1', 'type2', 'ip','amount']
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            df = get_two_count(df, trans, b[i], b[j])

    for x1 in ['op_type', 'op_mode', 'op_device', 'ip','channel','net_type']:
        for x2 in ['op_type', 'op_mode', 'op_device', 'ip','channel','net_type']:
            if x1 != x2:
                df = get_two_shuxing_count(df, op, x1, x2)
    print('operation_data is over!')
    for x1 in ['platform', 'tunnel_out', 'type1', 'type2', 'ip','amount']:
        for x2 in ['platform', 'tunnel_out', 'type1', 'type2', 'ip','amount']:
            if x1 != x2:
                df = get_two_shuxing_count(df, trans, x1, x2)
    print('transaction_data is over!')

    df = get_three_count(df, trans, 'platform', 'type1', 'tunnel_out')
    df = get_three_count(df, trans, 'platform', 'type2', 'tunnel_out')
    df = get_three_count(df, trans, 'amount', 'type2', 'ip')
    df = get_three_count(df, trans, 'amount', 'type1', 'ip')
    df = get_three_count(df, trans, 'amount', 'tunnel_out', 'ip')

    df = get_three_count(df, op, 'op_type', 'op_mode', 'op_device')
    df = get_three_count(df, op, 'ip', 'net_type', 'channel')
    df = get_three_count(df, op, 'ip_3', 'net_type', 'channel')
    df = get_three_count(df, op, 'ip', 'op_type', 'op_device')

    # fp = open("all_emb1.pkl", "rb+")
    # df = pickle.load(fp)
    #
    for col in tqdm(
            ['age', 'using_time', 'balance']):
        df[col + '_count'] = df[col].map(df[col].value_counts())
        df[col + '_count_ratio'] = df[col + '_count'] / df.shape[0]

    for col in tqdm(['product7_fail_cnt', 'level', 'acc_count', 'login_cnt_avg']):
        df[col + '_count'] = df[col].map(df[col].value_counts())

    df['id'] = df.index + 1
    cross_cols1 = ['age', 'using_time', 'city', 'balance', 'product7_fail_cnt', 'level', 'acc_count', 'login_cnt_avg']
    cross_cols = ['age', 'using_time', 'city', 'balance', 'acc_count', 'login_cnt_avg']
    for f in cross_cols:
        for col in cross_cols1:
            if col == f:
                continue
            print('------------------ {} {} ------------------'.format(f, col))
            df = df.merge(df[[f, col]].groupby(f, as_index=False)[col].agg({
                'cross_{}_{}_nunique'.format(f, col): 'nunique',
                'cross_{}_{}_ent'.format(f, col): lambda x: entropy(x.value_counts() / x.shape[0])  # 熵
            }), on=f, how='left')
            if 'cross_{}_{}_count'.format(f, col) not in df.columns.values:
                df = df.merge(df[[f, col, 'id']].groupby([f, col], as_index=False)['id'].agg({
                    'cross_{}_{}_count'.format(f, col): 'count'  # 共现次数
                }), on=[f, col], how='left')
            # if 'cross_{}_{}_count_ratio'.format(col, f) not in df.columns.values:
            #     df['cross_{}_{}_count_ratio'.format(col, f)] = df['cross_{}_{}_count'.format(f, col)] / df[
            #         f + '_count']  # 比例偏好
            if 'cross_{}_{}_count_ratio'.format(f, col) not in df.columns.values:
                df['cross_{}_{}_count_ratio'.format(f, col)] = df['cross_{}_{}_count'.format(f, col)] / df[
                    col + '_count']  # 比例偏好
            df['cross_{}_{}_nunique_ratio_{}_count'.format(f, col, f)] = df['cross_{}_{}_nunique'.format(f, col)] / \
                                                                         df[f + '_count']
            # df = reduce_mem(df)
    gc.collect()

    # fp = open("all_emb2.pkl", "rb+")
    # df = pickle.load(fp)


    df = get_time_frequence(df, op, 'op')
    df = gen_user_amount_frequence(df, trans)
    df = df.merge(gen_user_window_amount_features_hour(trans,6),on=['user'],how='left')
    for x in tqdm(['ip']):
        df = amt_label_frequence(df,trans,x)
    df = get_user_pay(df, trans)
    df = get_user_pay(df, op)

    trans_df = trans.sort_values(['user', 'days_diff', 'hour', 'minute'])

    for i in tqdm(['amount']):
        res = trans_df.groupby('user')[i].apply(lambda x: list(x)).reset_index()
        train_vec(res, i, 16, 'trans1')
        res = trans_df[trans_df['days_diff'] > 27].groupby('user')[i].apply(lambda x: list(x)).reset_index()
        train_vec(res, i, 10, 'trans2')
        res = trans_df[trans_df['days_diff'] > 23].groupby('user')[i].apply(lambda x: list(x)).reset_index()
        train_vec(res, i, 10, 'trans3')
        res = trans_df[(trans_df['hour'] >= 7) & (trans_df['hour'] <= 15)].groupby('user')[i].apply(
            lambda x: list(x)).reset_index()
        train_vec(res, i, 10, 'trans4')
        res = trans_df[trans_df['days_diff'] > 15].groupby('user')[i].apply(lambda x: list(x)).reset_index()
        train_vec(res, i, 10, 'trans5')

    for i in ['amount']:
        fp = open('trans1_' + i+"_emb.pkl", "rb+")
        res = pickle.load(fp)
        gc.collect()
        res.columns = ['user'] + ['1_emb_{}'.format(j) for j in range(16)]
        df = pd.merge(df, res, on='user', how='left')

        fp = open('trans2_' + i + "_emb.pkl", "rb+")  # 27
        res = pickle.load(fp)
        gc.collect()
        res.columns = ['user'] + ['2_emb_{}'.format(j) for j in range(10)]
        df = pd.merge(df, res, on='user', how='left')

        fp = open('trans3_' + i + "_emb.pkl", "rb+")  # 23
        res = pickle.load(fp)
        gc.collect()
        res.columns = ['user'] + ['3_emb_{}'.format(j) for j in range(10)]
        df = pd.merge(df, res, on='user', how='left')

        fp = open('trans4_' + i + "_emb.pkl", "rb+")  # 7-15
        res = pickle.load(fp)
        gc.collect()
        res.columns = ['user'] + ['4_emb_{}'.format(j) for j in range(10)]
        df = pd.merge(df, res, on='user', how='left')

        fp = open('trans5_' + i + "_emb.pkl", "rb+")  # 15
        res = pickle.load(fp)
        gc.collect()
        res.columns = ['user'] + ['5_emb_{}'.format(j) for j in range(10)]
        df = pd.merge(df, res, on='user', how='left')

    # LabelEncoder
    cat_cols = []
    for col in tqdm([f for f in df.select_dtypes('object').columns if f not in ['user']]):
        le = LabelEncoder()
        df[col].fillna('-1', inplace=True)
        df[col] = le.fit_transform(df[col])
        cat_cols.append(col)


    useless_features = ['every_morning_amt_gap_max_ip','ip_morning_gap_max_trans','ip_morning_gap_std_trans','ip_morning_gap_sum_trans',
                        'ip_morning_gap_min_trans','every_morning_amt_gap_sum_ip','every_morning_amt_gap_std_ip','every_morning_amt_min_ip',
                        'ip_morning_gap_mean_trans','every_morning_amt_gap_mean_ip','type2_of_nunique_platform ','every_morning_amt_gap_min_ip',
                        'money_frequence_morning_gap_mean','ip_morning_min_trans','cross_acc_count_city_nunique_ratio_acc_count_count',
                        'cross_product7_fail_cnt_city_nunique_ratio_product7_fail_cnt_count','cross_acc_count_balance_nunique_ratio_acc_count_count',
                        'cross_acc_count_balance_nunique','cross_login_cnt_avg_balance_nunique','cross_age_balance_nunique_ratio_age_count',
                        'cross_age_balance_nunique','cross_acc_count_level_nunique_ratio_acc_count_count','cross_login_cnt_avg_balance_nunique_ratio_login_cnt_avg_count',
                        'cross_using_time_balance_nunique','type2_of_nunique_platform','type2_of_nunique_platform',
                        'product7_fail_cnt_count', 'level_count', 'acc_count_count', 'login_cnt_avg']
    use_features = [col for col in df.columns if col not in useless_features]
    # df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    return df[use_features]

def lgb_model(train, target, test, k, random_):
    feats = [f for f in train.columns if f not in ['user', 'label']]
    print('Current num of features:', len(feats))
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_)
    oof_probs = np.zeros(train.shape[0])
    output_preds = 0
    offline_score = []
    feature_importance_df = pd.DataFrame()
    model = LGBMClassifier(objective='binary',
                           boosting_type='gbdt',
                           num_leaves=35,
                           max_depth=6,
                           learning_rate=0.01,
                           n_estimators=10000,
                           subsample=0.8,
                           feature_fraction=0.7,
                           reg_alpha=10,
                           reg_lambda=12,
                           random_state=random_,
                           is_unbalance=True,
                           metric='auc',
                           n_jobs=8,
                           bagging_freq=1)

    for i, (train_index, test_index) in enumerate(folds.split(train, target)):
        train_y, test_y = target[train_index], target[test_index]
        train_X, test_X = train[feats].iloc[train_index, :], train[feats].iloc[test_index, :]

        lgb_model = model.fit(train_X,
                              train_y,
                              eval_names=['train', 'valid'],
                              eval_set=[(train_X, train_y), (test_X, test_y)],
                              verbose=100,
                              # categorical_feature = [i for i in data if 'isre_' in i],
                              early_stopping_rounds=200)

        oof_probs[test_index] = lgb_model.predict_proba(test_X, num_iteration=lgb_model.best_iteration_)[:,1]
        output_preds += lgb_model.predict_proba(test[feats], num_iteration=lgb_model.best_iteration_)[:,1]/folds.n_splits
        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = lgb_model.feature_importances_
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('feature importance:')
    print(feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False))

    return output_preds, oof_probs

# def xgb_model(train,target,test,k,random_):
#     feats = [f for f in train.columns if f not in ['user', 'label']]
#     print('Current num of features:', len(feats))
#     folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_)
#     oof_probs = np.zeros(train.shape[0])
#     output_preds = 0
#     for i, (train_index, test_index) in enumerate(folds.split(train, target)):
#         train_y, test_y = target[train_index], target[test_index]
#         train_X, test_X = train[feats].iloc[train_index, :], train[feats].iloc[test_index, :]
#         clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.01,eval_metric='auc',
#                                 n_estimators=6000,nthread=10,subsample=0.8,colsample_bytree=0.7,
#                                 reg_alpha=10,reg_lambda=12,scale_pos_weight=3,objective='binary:logistic')
#         clf.fit(train_X, train_y,
#                 eval_set=[(train_X, train_y), (test_X, test_y)],
#                 early_stopping_rounds=200,
#                 verbose=100)
#
#         oof_probs[test_index] = clf.predict_proba(test_X, clf.best_iteration)[:,1]
#         output_preds += clf.predict_proba(test[feats], clf.best_iteration)[:,1]/folds.n_splits
#
#
#     return output_preds, oof_probs




if __name__ == '__main__':
    a = 0
    num = 5
    times = 5
    index = 'lgb_{}_{}'.format(num,times)
    for i in range(num):
        DATA_PATH = 'data/'
        print('读取数据...')
        data, op_df, trans_df = data_preprocess(DATA_PATH=DATA_PATH)

        print('开始特征工程...')
        data = gen_features(data, op_df, trans_df)
        data['city_level'] = data['city'].map(str) + '_' + data['level'].map(str)
        data['city_balance_avg'] = data['city'].map(str) + '_' + data['balance_avg'].map(str)

        print('开始模型训练...')
        train = data[~data['label'].isnull()].copy()
        target = train['label']
        test = data[data['label'].isnull()].copy()

        target_encode_cols = ['province', 'city', 'city_level', 'city_balance_avg']

        train, test = kfold_stats_feature(train, test, target_encode_cols, 5, i)
        train.drop(['city_level', 'city_balance_avg'], axis=1, inplace=True)
        test.drop(['city_level', 'city_balance_avg'], axis=1, inplace=True)

        drop_train = train.T.drop_duplicates().T
        drop_test = test.T.drop_duplicates().T

        features = [i for i in drop_train.columns if i in drop_test.columns]
        lgb_preds, lgb_oof = lgb_model(train=train[features+['label']], target=target, test=test[features+['label']], k=5, random_= i)

        # index = 'xgb'
        # lgb_preds, lgb_oof = xgb_model(train=train[features + ['label']], target=target, test=test[features + ['label']],
        #                                k=5,random_ = i)
        a += lgb_preds / num
        sub_df = test[['user']].copy()
        sub_df['prob'] = a
        sub_df.to_csv('submission/{}_sub.csv'.format(index), index=False)
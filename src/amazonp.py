import os, sys, gzip
import argparse, random, pickle, nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import count
from stanfordcorenlp import StanfordCoreNLP
from operator import itemgetter
from collections import defaultdict
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer


# 转为df
def get_df(path_s,path_t):
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            name = b'"verified": \"true\",'
            l = l.replace(b'"verified": true,', bytes(name))
            name = b'"verified": \"false\",'
            l = l.replace(b'"verified": false,', bytes(name))
            yield eval(l)

    def get_raw_df(path):
        df = {}
        for i, d in tqdm(enumerate(parse(path)), ascii=True):
            if "" not in d.values():
                df[i] = d
        df = pd.DataFrame.from_dict(df, orient='index')
        df = df[["reviewerID", "asin", "reviewText", "overall"]]  # user,item, text, score
        return df

    csv_path_s = path_s.replace('.json.gz', '.csv')
    csv_path_t = path_t.replace('.json.gz', '.csv')

    if os.path.exists(csv_path_s) and os.path.exists(csv_path_t):
        df_s = pd.read_csv(csv_path_s)
        df_t = pd.read_csv(csv_path_t)
        print('Load raw data from %s.' % csv_path_s)
        print('Load raw data from %s.' % csv_path_t)
    else:
        df_s = get_raw_df(path_s)
        df_t = get_raw_df(path_t)

        df_s.to_csv(csv_path_s, index=False)
        df_t.to_csv(csv_path_t, index=False)
        print('Build raw data to %s.' % csv_path_s)
        print('Build raw data to %s.' % csv_path_t)

    return df_s, df_t


# 过滤交互数量
def filterout(df_s, df_t, thre_i, thre_u):
    index_s = df_s[["overall", "asin"]].groupby('asin').count() >= thre_i
    index_t = df_t[["overall", "asin"]].groupby('asin').count() >= thre_i # 物品至少有x条交互
    item_s = set(index_s[index_s['overall'] == True].index)
    item_t = set(index_t[index_t['overall'] == True].index)
    df_s = df_s[df_s['asin'].isin(item_s)]
    df_t = df_t[df_t['asin'].isin(item_t)]

    index_s = df_s[["overall", "reviewerID"]].groupby('reviewerID').count() >= thre_u # 至少有x条交互
    index_t = df_t[["overall", "reviewerID"]].groupby('reviewerID').count() >= thre_u
    user_s = set(index_s[index_s['overall'] == True].index)
    user_t = set(index_t[index_t['overall'] == True].index)
    df_s = df_s[df_s['reviewerID'].isin(user_s)]
    df_t = df_t[df_t['reviewerID'].isin(user_t)]

    return df_s, df_t

# 重新编码
def convert_idx(df_s, df_t):
    uiterator = count(1) # start==1
    udict = defaultdict(lambda: next(uiterator)) # https://zhuanlan.zhihu.com/p/443710807
    [udict[user] for user in df_s["reviewerID"].tolist() + df_t["reviewerID"].tolist()] # 联合编码
    iiterator_s = count(1)
    idict_s = defaultdict(lambda: next(iiterator_s))
    [idict_s[item] for item in df_s["asin"]]
    iiterator_t = count(1)
    idict_t = defaultdict(lambda: next(iiterator_t))
    [idict_t[item] for item in df_t["asin"]]

    df_s['uid'] = df_s['reviewerID'].map(lambda x: udict[x]) # old:new
    df_t['uid'] = df_t['reviewerID'].map(lambda x: udict[x])
    df_s['iid'] = df_s['asin'].map(lambda x: idict_s[x])
    df_t['iid'] = df_t['asin'].map(lambda x: idict_t[x])

    user_set_s = set(df_s['uid'])
    item_set_s = set(df_s['iid'])
    user_set_t = set(df_t['uid'])
    item_set_t = set(df_t['iid'])
    overlap_user_set = user_set_s & user_set_t
    all_user_set = user_set_s | user_set_t

    assert len(item_set_s) == len(idict_s)
    assert len(item_set_t) == len(idict_t)

    user_num_s, item_num_s, user_num_t, item_num_t, overlap_num_user, user_num = \
        len(user_set_s), len(item_set_s), len(user_set_t), len(item_set_t), len(overlap_user_set), len(all_user_set)

    print('Source domain users %d, items %d, ratings %d.' % (user_num_s, item_num_s, len(df_s)))
    print('Target domain users %d, items %d, ratings %d.' % (user_num_t, item_num_t, len(df_t)))
    print('Overlapping users %d (%.3f%%, %.3f%%).' % (
        overlap_num_user, overlap_num_user/user_num_s*100, overlap_num_user/user_num_t*100))

    return dict(udict), dict(idict_s), dict(idict_t), overlap_user_set, df_s, df_t


# 获取review的emb

#get chinese nlp
nlp = StanfordCoreNLP('/dfs/data/miniconda3/envs/ORec/lib/python3.8/site-packages/stanfordcorenlp/stanford-corenlp-full-2022-07-25',lang='zh')
nltk.download('stopwords')
root_p = r"/dfs/data/ORec/data/"

def is_chinese(uchar):
    """is this a chinese word?"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """is this unicode a number?"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """is this unicode an English word?"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False

def format_str(content, lag):
    content_str = ''
    if lag == 0:  # English
        for i in content:
            if is_alphabet(i):
                content_str = content_str + i
    if lag == 1:  # Chinese
        for i in content:
            if is_chinese(i):
                content_str = content_str + i
    if lag == 2:  # Number
        for i in content:
            if is_number(i):
                content_str = content_str + i
    return content_str

def merge(row):
    """review_text处理"""
    str_cleaned = ""
    if row["reviewText"] != "None":
        str_cleaned = format_str(row["reviewText"], 0)
    if str_cleaned=='':
       return ""
    else:
        str_cleaned = str_cleaned[:20000] # 太长会decode失败
        words = nlp.word_tokenize(str_cleaned)
        return words


def doc_emb_cop(u_file, i_file, vec_num, domain="shop"):
    """这里使用user/item合计"""
    print("Start!")
    documents = u_file['reviewText'].values.tolist() + \
                i_file['reviewText'].values.tolist()
    frequency = defaultdict(int)
    for text in documents:
        for token in text:
            frequency[token] += 1
    # 大于1的数据
    texts = [[token for token in text if frequency[token] > 1]
             for text in documents]
    # train the model
    documents = [TaggedDocument(doc, [int(i)]) for i, doc in enumerate(documents)]
    # documents: Users + Movies
    docs = documents
    model = Doc2Vec(docs, vector_size=vec_num, window=2, min_count=5, negative=5, workers=6)
    model.train(docs, total_examples=model.corpus_count, epochs=50)
    model.save(root_p + "amazon_" + domain + "/Doc2vec_amazon_%s_VSize%02d.model" % (domain,vec_num))
    vectors = model.docvecs.vectors_docs
    print("End!")



def review_info(df_s, df_t, u_unique_lis, vec_num):
    # source: u_info
    df = pd.DataFrame({"uid": u_unique_lis})
    df_s_u = df_s.groupby('uid')['reviewText'].apply(list).to_frame().reset_index(drop=False)
    df_s_u = pd.merge(df, df_s_u, on='uid', how='left')[['uid', 'reviewText']]
    df_s_u.sort_values(by="uid", inplace=True, ascending=True)
    df_s_u['reviewText'] = df_s_u['reviewText'].fillna("No text.")
    df_s_u["reviewText"] = df_s_u.apply(merge, axis=1)

    # source: i_info
    df_s_i = df_s.groupby('iid')['reviewText'].apply(list).to_frame().reset_index(drop=False)
    df_s_i.sort_values(by="iid", inplace=True, ascending=True)
    df_s_i['reviewText'] = df_s_i['reviewText'].fillna("无")
    df_s_i["reviewText"] = df_s_i.apply(merge, axis=1)

    doc_emb_cop(df_s_u, df_s_i, vec_num, domain="movie")


    # target: u_info
    df = pd.DataFrame({"uid": u_unique_lis})
    df_t_u = df_t.groupby('uid')['reviewText'].apply(list).to_frame().reset_index(drop=False)
    df_t_u = pd.merge(df, df_t_u, on='uid', how='left')[['uid', 'reviewText']]
    df_t_u.sort_values(by="uid", inplace=True, ascending=True)
    df_t_u['reviewText'] = df_t_u['reviewText'].fillna("No text.")
    df_t_u["reviewText"] = df_t_u.apply(merge, axis=1)

    # target: i_info
    df_t_i = df_t.groupby('iid')['reviewText'].apply(list).to_frame().reset_index(drop=False)
    df_t_i.sort_values(by="iid", inplace=True, ascending=True)
    df_t_i['reviewText'] = df_t_i['reviewText'].fillna("No text.")
    df_t_i["reviewText"] = df_t_i.apply(merge, axis=1)

    doc_emb_cop(df_t_u, df_t_i, vec_num, domain="music")





if __name__ == '__main__':
    # path_s = r"E:\Code\Pycharm\ORec\data\amazon_raw\Movies_and_TV.json.gz"
    # path_t = r"E:\Code\Pycharm\ORec\data\amazon_raw\CDs_and_Vinyl.json.gz"
    # df_s, df_t = get_df(path_s, path_t)
    domain_A, domain_B = 'movie', 'music'
    df_s = pd.read_csv(r"E:\Code\Pycharm\ORec\data\amazon_raw\Movies_and_TV.csv")
    df_t = pd.read_csv(r"E:\Code\Pycharm\ORec\data\amazon_raw\CDs_and_Vinyl.csv")

    df_s, df_t = filterout(df_s, df_t, thre_i=30, thre_u=30)  # 按交互数过滤
    udict, idict_s, idict_t, overlap_user_set, df_s, df_t = convert_idx(df_s, df_t) # old_new

    df_s.to_csv(root_p+'amazon_'+domain_A + r"\ratings_p.csv")
    df_t.to_csv(root_p+'amazon_'+domain_B + r"\ratings_p.csv")


    review_info(df_s, df_t, list(udict.values()), vec_num=64)

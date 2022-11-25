import numpy as np
import pymongo
import pandas as pd
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from string import punctuation
import tensorflow_hub as hub

load_dotenv()

fields = {"id", "instagram.username", "instagram.biography", "instagram.category_name",
          "instagram.post_data.posts.caption", "instagram.post_data.posts.like_to_follower_ratio",
          "instagram.follower_count", "instagram.is_verified", "instagram.external_url",
          "instagram.reels", "score"}
filters = {"instagram.biography": {"$exists": "true"},
           "instagram.post_data.posts.caption": {"$exists": "true"}}


def setup_connection(collection):
    path = os.getenv('MONGO_PRODUCTION_URL')
   
    client = pymongo.MongoClient(path)
    db = client["ask_emma"]
    return db[collection]


def load_data():
    collection = setup_connection('ml')
    cursor = collection.find(filters, fields)
    return pd.json_normalize(cursor)


def parse_likes(posts):
    size = len(posts)
    if size == 0:
        return 0
    total = 0

    for post in posts:
        total += float(post['like_to_follower_ratio'])

    engagement = total / size
    if engagement > 10:
        engagement = 10
    return engagement


def concat_captions(posts):
    captions = ''
    for post in posts:
        captions += ' ' + post['caption']
    return captions


def get_domain(url):
    domain = urlparse(url).netloc
    filter_domain = '.'.join(domain.split('.')[-2:])
    return filter_domain


def parse_username(username):
    username = username.lower()
    good_words = ['esthetician', 'makeup', 'artist', 'mua', 'glowing', 'glow', 'skincare', 'skin']
    score = 0.7
    for word in good_words:
        if word in username:
            score = 1
            continue

    if any(p in username for p in punctuation):
        score = 0

    return score


def parse_bio(bio):
    bio_words = ['newsletter', 'substack', 'podcast']
    bio = bio.lower()
    for word in bio_words:
        if word in bio:
            return 1
    return 0


def parse_youtube(bio, domain):
    if 'you' in domain:
        return 1
    bio_words = ['youtube', 'yt']
    bio = bio.lower()
    for word in bio_words:
        if word in bio:
            return 1
    return 0


def parse_tiktok(bio, domain):
    bio = bio.lower()
    if 'tiktok' in domain or 'tiktok' in bio:
        return 1
    return 0


def get_domain_score(d):
    quality_links = ['linktr.ee', 'msha.ke', 'beacons.ai', 'hoo.be', 'linkin.biov', 'beacons.page', 'komi.io', 'direct.me',
                    'flow.page', 'lnk.bio', 'solo.to', 'linkr.bio', 'linkpop.com', 'instabio.cc', 'koji.to', 'liinks.co',
                    'stan.store', 'withkoji.com', 'linktree.com', 'pico.link', 'likeshop.me']
    competitor = ['amazon.com', 'shopltk.com', 'liketoknow.it', 'glnk.io']
    average_links = ['campsite.bio', 'zez.am', 'snipfeed.co']
    bad_links = ['bit.ly', 'co.uk', 'vsco.co', 'canva.site', 'as.me', 'facebook.com', 'thewallgroup.com',
                'depop.com', 'radiatebyamanda.com', 'com.au']
    shopmy = ['shoplist.us', 'shopmy.us', 'shopmyshelf.us', 'myshlf.us']

    if d in quality_links or d in shopmy:
        return 1
    elif d in competitor:
        return 1.25
    elif d in average_links:
        return 0.5
    elif d in bad_links:
        return 0
    return 0


def prepare_features(df):
    df = df.fillna('0')
    df['engagement'] = df.apply(lambda x: parse_likes(x['instagram.post_data.posts']), axis=1)
    df['captions'] = df.apply(lambda x: concat_captions(x['instagram.post_data.posts']), axis=1)
    df['domain'] = df.apply(lambda x: get_domain(x['instagram.external_url']), axis=1)
    df['domain_score'] = df.apply(lambda x: get_domain_score(x['domain']), axis=1)
    df['username_score'] = df.apply(lambda x: parse_username(x['instagram.username']), axis=1)
    df['bio_key_words'] = df.apply(lambda x: parse_bio(x['instagram.biography']), axis=1)
    df['youtuber'] = df.apply(lambda x: parse_youtube(x['instagram.biography'], x['domain']), axis=1)
    df['tiktoker'] = df.apply(lambda x: parse_tiktok(x['instagram.biography'], x['domain']), axis=1)
    return df


def write_data(df, collection):
    collection = setup_connection(collection)
    collection.insert_many(df.to_dict('records'))


def apply_transformations(df):
    df['instagram.is_verified'].replace(list(df['instagram.is_verified'].unique()),
                                        np.arange(len(list(df['instagram.is_verified'].unique()))), inplace=True)

    conditions = [
        (df['instagram.follower_count'].between(0, 1000)),
        (df['instagram.follower_count'].between(1000, 10000)),
        (df['instagram.follower_count'].between(10000, 100000)),
        (df['instagram.follower_count'] >= 100000)
    ]
    values = [1, 2, 3, 4]

    df['instagram.follower_count'] = np.select(conditions, values)
    df['highlights'] = df['instagram.reels'].map(lambda x: ' '.join(map(str, x)))
    df['highlights'] = df['highlights'].replace('0', '', regex=True)
    df['instagram.biography'] += df['highlights']

    df = df.dropna()
    return df


def embed_data(df_subset, embed):
    bios = pd.DataFrame(embed(df_subset['instagram.biography']).numpy())
    pca = PCA(n_components=12)
    principal_components = pca.fit_transform(bios)
    bios = pd.DataFrame(principal_components)
    df_subset = df_subset.drop(columns=['instagram.biography'])
    bios = bios.add_suffix('_bio')
    idx = df_subset.index
    df_subset = df_subset.reset_index(drop=True)
    df_subset = pd.concat([df_subset, bios], axis=1)
    # df_subset = df_subset.astype(float)
    df_subset = df_subset.set_index(idx)
    return df_subset


def apply_scaler(df_subset):
    scaler = MinMaxScaler()
    return scaler.fit_transform(df_subset)


def apply_clustering(scaled_data, n):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(scaled_data)
    labels = kmeans.predict(scaled_data)
    return labels


def apply_weights(scaled_data, cols):
    weight_follower_count = 1
    weight_engagement = 1.2
    weight_domain = 1
    weight_blue_tick = 1.3
    weight_bio_key_words = 0.5
    weight_username_score = 0.3
    weight_youtube = 1
    weight_tiktok = 0.7
    scaled_data[:, cols.index('instagram.follower_count')] = scaled_data[:, cols.index(
        'instagram.follower_count')] * weight_follower_count
    scaled_data[:, cols.index('engagement')] = scaled_data[:, cols.index('engagement')] * weight_engagement
    scaled_data[:, cols.index('domain_score')] = scaled_data[:, cols.index('domain_score')] * weight_domain
    scaled_data[:, cols.index('instagram.is_verified')] = scaled_data[:,
                                                          cols.index('instagram.is_verified')] * weight_blue_tick
    scaled_data[:, cols.index('bio_key_words')] = scaled_data[:, cols.index('bio_key_words')] * weight_bio_key_words
    scaled_data[:, cols.index('username_score')] = scaled_data[:, cols.index('username_score')] * weight_username_score
    scaled_data[:, cols.index('youtuber')] = scaled_data[:, cols.index('youtuber')] * weight_youtube
    scaled_data[:, cols.index('tiktoker')] = scaled_data[:, cols.index('tiktoker')] * weight_tiktok
    return scaled_data


def get_all_buckets(df):
    df_buckets = pd.DataFrame(
        columns=['bucket', 'similar_usernames', 'similar_ids', 'follower_counts', 'username_score', 'domain_scores',
                 'engagements', 'is_verified', 'youtuber', 'tiktoker', 'bio_key_words', 'mean_score'])
    df_buckets['bucket'] = range(n)
    for i in range(n):
        df_buckets['similar_usernames'][i] = list(df['instagram.username'][list(df.index[df['cluster'] == i])])
        df_buckets['similar_ids'][i] = list(df['id'][list(df.index[df['cluster'] == i])])
        df_buckets['follower_counts'][i] = list(df['instagram.follower_count'][list(df.index[df['cluster'] == i])])
        df_buckets['engagements'][i] = list(df['engagement'][list(df.index[df['cluster'] == i])])
        df_buckets['domain_scores'][i] = list(df['domain_score'][list(df.index[df['cluster'] == i])])
        df_buckets['is_verified'][i] = list(df['instagram.is_verified'][list(df.index[df['cluster'] == i])])
        df_buckets['youtuber'][i] = list(df['youtuber'][list(df.index[df['cluster'] == i])])
        df_buckets['username_score'][i] = list(df['username_score'][list(df.index[df['cluster'] == i])])
        df_buckets['bio_key_words'][i] = list(df['bio_key_words'][list(df.index[df['cluster'] == i])])
        df_buckets['tiktoker'][i] = list(df['tiktoker'][list(df.index[df['cluster'] == i])])
        df_buckets['mean_score'][i] = np.mean(df['score'][list(df.index[df['cluster'] == i])].astype(float))
    return df_buckets


def get_urgency(bucket):
    scale = {1: 'F', 2: 'D', 3: 'B', 4: 'A', 5: 'A+'}
    if bucket['avg_is_verified'] > 0.9:
        return scale[5]
    if bucket['avg_youtuber'] > 0.9:
        return scale[4]
    if bucket['avg_follower'] > 3.5:
        return scale[4]
    if bucket['avg_domain_scores'] >= 1:
        return scale[4]
    if bucket['avg_engagement'] > 0.1:
        return scale[4]
    if bucket['avg_tiktoker'] > 0.8:
        return scale[3]
    if bucket['avg_is_verified'] + bucket['avg_domain_scores'] == 0:
        return scale[1]
    return scale[2]


def get_relevance(bucket):
    scale = {1: 'important', 2: 'expert', 3: 'affliate', 4: 'na'}
    if bucket['avg_is_verified'] > 0.9:
        return scale[1]
    if bucket['avg_youtuber'] > 0.9:
        return scale[2]
    if bucket['avg_follower'] > 3.5:
        return scale[2]
    if bucket['avg_domain_scores'] >= 1:
        return scale[2]
    if bucket['avg_engagement'] > 0.1:
        return scale[2]
    if bucket['avg_tiktoker'] > 0.8:
        return scale[3]
    if bucket['avg_is_verified'] + bucket['avg_domain_scores'] == 0:
        return scale[4]
    return scale[4]


def assign_tags(df_buckets):
    df_buckets['avg_follower'] = df_buckets['follower_counts'].map(lambda x: sum(x)/len(x))
    df_buckets['avg_engagement'] = df_buckets['engagements'].map(lambda x: sum(x)/len(x))
    df_buckets['avg_domain_scores'] = df_buckets['domain_scores'].map(lambda x: sum(x) / len(x))
    df_buckets['avg_is_verified'] = df_buckets['is_verified'].map(lambda x: sum(x) / len(x))
    df_buckets['avg_youtuber'] = df_buckets['youtuber'].map(lambda x: sum(x) / len(x))
    df_buckets['avg_tiktoker'] = df_buckets['tiktoker'].map(lambda x: sum(x) / len(x))
    df_buckets['tag1'] = df_buckets.apply(lambda x: get_urgency(x), axis=1)
    df_buckets['tag2'] = df_buckets.apply(lambda x: get_relevance(x), axis=1)
    return df_buckets


if __name__ == '__main__':
    df = load_data()
    df = prepare_features(df)
    df = apply_transformations(df)

    df_subset = df[['instagram.biography', 'instagram.follower_count', 'engagement', 'domain_score',
                    'instagram.is_verified', 'bio_key_words', 'username_score', 'youtuber', 'tiktoker']]
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    df_subset = embed_data(df_subset, embed)
    scaled_data = apply_scaler(df_subset)

    n = 90
    cols = df_subset.columns.to_list()
    scaled_data = apply_weights(scaled_data, cols)
    clusters = apply_clustering(scaled_data, n)
    df['cluster'] = clusters

    user_bucket = df[['id', 'instagram.username', 'cluster', 'score']]
    # write_data(user_bucket, 'ml_outputs')

    df_buckets = get_all_buckets(df)
    df_buckets = assign_tags(df_buckets)
    df_mongo = df_buckets[['bucket', 'similar_usernames', 'tag1', 'tag2', 'similar_ids']]
    # write_data(df_mongo, 'ml_buckets')
    print('end')

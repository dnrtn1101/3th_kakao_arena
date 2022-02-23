import argparse
import pickle
import collections

import numpy as np
import pandas as pd

import sentencepiece as spm
from gensim.models import fasttext

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

class TitleEmbedding:
    def __init__(self,
                 vocab_size=3000):
        self.vocab_size = vocab_size
        self.prefix = f'vs_({vocab_size})'

    def fit(self, df):
        df = df.copy()
        # Cleaning the title
        df['title'] = df['plylst_title'].apply(lambda x: self.title_cleaning(x))

        # Saving the text file to be used as input to the sentencepiece
        self.__write_title(df['title'].tolist())

        # Training the sentencepiece
        self.sp = self.__train_spm()

        # Claening the title again by using the sentencepiece model
        df['title'] = df['title'].apply(lambda x: x if len(x) == 0 else self.cleaning_sentence(x))

        # Trainig the fasttext model
        tf_config = {'min_count': 5,
                     'size': 128,
                     'sg': 1,
                     'batch_words': 10000,
                     'iter': 20,
                     'window': 5,
                     'seed': 25,
                     'word_ngrams': 1}

        self.embedding_model = fasttext.FastText(sentences=list(map(lambda x: x.split(' '), df['title'].tolist())),
                                                 **tf_config)

        pad_vectors = np.zeros_like(self.embedding_model.wv.vectors[0])
        self.embedding_model.wv.vectors = np.insert(self.embedding_model.wv.vectors, 0, pad_vectors, axis=0)
        self.embedding_model.wv.index2word.insert(0, '<pad>')

    def __train_spm(self):
        # Train
        templates = f'--input=./output/titles.txt --model_prefix={self.prefix} --vocab_size={self.vocab_size}'
        spm.SentencePieceTrainer.Train(templates)

        # Loading the model and return
        sp = spm.SentencePieceProcessor()
        sp.Load(f'{self.prefix}.model')
        return sp

    def title_cleaning(self, title):
        return ''.join([s for s in title if (s.isalnum() or s == ' ')])

    def __write_title(self, titles):
        with open('./output/titles.txt', 'w', encoding='utf8') as f:
            for title in titles:
                f.write(f'{title}\n')

    def cleaning_sentence(self, sentence):
        ret = []
        tokens = self.sp.encode_as_pieces(sentence)
        while len(tokens) > 0:
            if tokens[0] == '▁':
                ret.append(tokens[1])
                sentence = self.sp.decode_pieces(tokens[2:])
            else:
                ret.append(' ' + tokens[0][1:])
                sentence = self.sp.decode_pieces(tokens[1:])
            tokens = self.sp.encode_as_pieces(sentence)

        ret = ''.join(ret)
        if len(ret) == 0:
            return ret

        elif ret[0] == ' ':
            ret = ret[1:]

        return ret

def create_item_to_id(o_dict, min_count, start_from):
    id_list = list(o_dict.keys())
    count_list = list(o_dict.values())

    if min_count > 1:
        remove_from = count_list.index(min_count - 1)
        del id_list[remove_from:]
        del count_list[remove_from:]

    item_to_id = dict(zip(id_list, range(start_from, start_from + len(id_list))))

    return item_to_id

def title_to_index(title, embedding_model, max_len=20):
    ret = []
    for word in title:
        try:
            idx = embedding_model.wv.vocab[word].index + 1
        except:
            idx = 0
        ret.append(idx)
    length = len(ret)
    if length > max_len:
        ret = ret[:max_len]
    else:
        ret += (max_len - length) * [0]
    return ret

def preprocess(args):
    data_path = args.data_path
    min_count_songs = args.min_count_songs
    min_count_tags = args.min_count_tags

    # Loading data and simplifying a field name.
    train = pd.read_json(f'{data_path}/train.json')
    val = pd.read_json(f'{data_path}/val.json')
    test = pd.read_json(f'{data_path}/test.json')

    songmeta = pd.read_json(f'{data_path}/song_meta.json')
    songmeta = songmeta.rename(columns={'id': 'songs',
                                        'song_gn_dtl_gnr_basket': 'genres'})

    # Building custom validation set.
    train_index, val_index = train_test_split(train.index, test_size=0.2, random_state=1)

    # marking and concatenating train, custom validation, val and test data.
    train.loc[train_index, 'dataset'] = 0
    train.loc[val_index, 'dataset'] = 1
    val['dataset'] = 2
    test['dataset'] = 3

    full = pd.concat([train, val, test], axis=0, ignore_index=True)
    del train, val, test

    # Creating time features.
    full['year'] = full['updt_date'].apply(lambda x: int(x.split('-')[0]))
    full['year'] = minmax_scale(full['year'])
    full['month'] = full['updt_date'].apply(lambda x: int(x.split('-')[1]))
    full['month'] = minmax_scale(full['month'])
    del full['updt_date'], full['like_cnt']

    # Building the song-to-index mapping and tag-to-index ============================================== #
    song_counter = collections.Counter(full[full['dataset'] == 0].explode('songs')['songs'].values)
    o_dict = collections.OrderedDict(song_counter.most_common())
    song_to_id = create_item_to_id(o_dict, min_count_songs, 0)

    tag_counter = collections.Counter(full[full['dataset'] == 0].explode('tags')['tags'].values)
    o_dict = collections.OrderedDict(tag_counter.most_common())
    tag_to_id = create_item_to_id(o_dict, min_count_tags, len(song_to_id))

    # Removing songs and tags under min count.
    full['songs'] = full['songs'].apply(lambda x: [y for y in x if y in song_to_id.keys()])
    full['tags'] = full['tags'].apply(lambda x: [y for y in x if y in tag_to_id.keys()])
    # ================================================================================================== #
    # Exploding the 'songs' column to merge the genres that each playlist contains.======================#
    full_exploded_songs = full.explode('songs')
    songmeta_exploded_gnr = songmeta.explode('genres')
    full_exploded_songs = full_exploded_songs.merge(right=songmeta_exploded_gnr[['songs', 'genres']], on='songs',
                                                    how='left')

    full_gnr = full_exploded_songs.groupby('id')['genres'].unique().reset_index()
    full = full.merge(right=full_gnr, on='id', how='left')
    del full_exploded_songs, songmeta_exploded_gnr, full_gnr
    # ================================================================================================== #
    # Building the gnr-to-index mapping and song-to-genres ============================================= #
    gnr_counter = collections.Counter(full[full['dataset'] == 0].explode('genres')['genres'].values)
    o_dict = collections.OrderedDict(gnr_counter.most_common())
    gnr_to_id = create_item_to_id(o_dict, 1, len(song_to_id) + len(tag_to_id))

    songmeta['genres'] = songmeta['genres'].apply(lambda x: [gnr_to_id[y] for y in x if y in gnr_to_id.keys()])
    song_to_genres = songmeta.loc[song_to_id.keys(), ['songs', 'genres']]
    song_to_genres['songs'] = song_to_genres['songs'].apply(lambda x: song_to_id[x])
    song_to_genres = dict(song_to_genres.values)
    # ================================================================================================== #
    # Concatenate the three mapping and saving.========================================================= #
    id_mapping = dict(song_to_id, **tag_to_id)
    id_mapping.update(gnr_to_id)

    mapping = dict()
    mapping['mapping'] = id_mapping
    mapping['inverse_mapping'] = dict(zip(mapping['mapping'].values(), mapping['mapping'].keys()))

    mapping['num_songs'] = len(song_to_id)
    mapping['num_tags'] = len(tag_to_id)
    mapping['num_genres'] = len(gnr_to_id)
    mapping['song_to_genres'] = song_to_genres

    with open(f'./output/mapping(0725_{min_count_songs}_{min_count_tags}).pickle', 'wb') as f:
        pickle.dump(mapping, f)
    # ================================================================================================== #
    # mapping each item to the corresponding index.
    full['songs'] = full['songs'].apply(lambda x: [song_to_id[y] for y in x if y in song_to_id.keys()])
    full['tags'] = full['tags'].apply(lambda x: [tag_to_id[y] for y in x if y in tag_to_id.keys()])
    full['genres'] = full['genres'].apply(lambda x: [gnr_to_id[y] for y in x if y in gnr_to_id.keys()])

    # Make 'my_val' similar to the test data
    full['masked_songs'] = full['songs'].copy()
    full['masked_tags'] = full['tags'].copy()

    full.loc[full['dataset'] == 1, 'masked_songs'] = full.loc[full['dataset'] == 1, 'songs'].apply(
        lambda x: np.random.choice(x, int(len(x) * 0.5), replace=False))
    full.loc[full['dataset'] == 1, 'masked_songs'] = full.loc[full['dataset'] == 1, 'masked_songs'].apply(
        lambda x: x if np.random.choice([0, 1], p=[0.2, 0.8]) == 1 else [])
    full.loc[full['dataset'] == 1, 'masked_tags'] = full.loc[full['dataset'] == 1, 'tags'].apply(
        lambda x: np.random.choice(x, int(len(x) * 0.5), replace=False))
    full.loc[full['dataset'] == 1, 'masked_tags'] = full.loc[full['dataset'] == 1, 'masked_tags'].apply(
        lambda x: x if np.random.choice([0, 1], p=[0.34, 0.66]) == 1 else [])
    # ================================================================================================== #
    ### Part. Title preprocessing
    full['len_title'] = full['plylst_title'].apply(lambda x: len(x))
    condition = ((full['dataset'] == 2) | (full['dataset'] == 3)) & (full['len_title']>0)
    embedding = TitleEmbedding(vocab_size=3000)
    embedding.fit(full[condition])
    full['title'] = full['plylst_title'].apply(lambda x: embedding.title_cleaning(x))
    full['title'] = full['title'].apply(lambda x: embedding.cleaning_sentence(x))
    full['title'] = full['title'].apply(lambda x: title_to_index(x, embedding_model = embedding.embedding_model, max_len=10))

    with open(f'./output/embedding.pickle', 'wb') as f:
        pickle.dump(embedding.embedding_model.wv.vectors, f)

    del full['len_title'], full['plylst_title']
    # ================================================================================================== #
    ### Part. Saving data
    full.to_json(f'./output/full(0725_{min_count_songs}_{min_count_tags}).json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--min_count_songs', type=int, default=4)
    parser.add_argument('--min_count_tags', type=int, default=2)

    args = parser.parse_args()

    preprocess(args)

if __name__ == '__main__':
    main()
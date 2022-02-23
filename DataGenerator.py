import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 df,
                 song_to_genres,
                 dim,
                 batch_size=4,
                 mode='train',
                 mask=False,
                 shuffle=True):
        super(DataGenerator, self).__init__()
        self.df = df.copy()
        self.song_to_genres=song_to_genres
        self.dim = dim
        self.batch_size = batch_size
        self.mode = mode
        self.mask = mask
        self.shuffle = shuffle

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.array(self.df.index)
        if self.mask==True:
            self.df['masked_songs'] = self.df['songs'].apply(lambda x: list(np.random.choice(x, size=int(len(x)*0.5), replace=False)))
            self.df['masked_songs'] = self.df['masked_songs'].apply(lambda x: x if np.random.choice([0, 1], p=[0.2, 0.8]) == 1 else [])
            self.df['masked_tags'] = self.df['tags'].apply(lambda x: list(np.random.choice(x, size=int(len(x) * 0.5), replace=False)))
            self.df['masked_tags'] = self.df['masked_tags'].apply(lambda x: x if np.random.choice([0, 1], p=[0.34, 0.66]) == 1 else [])
            self.df['masked_genres'] = self.df['masked_songs'].apply(lambda x: [] if len(x) == 0 else list(set(np.hstack([self.song_to_genres[y] for y in x]))))
            self.df['masked_genres'] = self.df['masked_genres'].apply(lambda x: list(map(int, x)))

        if self.shuffle==True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X = np.zeros((self.batch_size, self.dim), dtype=np.float32)
        y = np.zeros((self.batch_size, self.dim), dtype=np.float32)
        for i, index in enumerate(batch_indexes):
            songs = self.df.loc[index, 'songs']
            tags = self.df.loc[index,'tags']
            genres = self.df.loc[index,'genres']
            year = self.df.loc[index, 'year']
            month = self.df.loc[index, 'month']

            y[i, songs] = 1
            y[i, tags] = 1
            y[i, genres] = 1
            y[i, -2] = year
            y[i, -1] = month

            X[i, -2] = year
            X[i, -1] = month
            if self.mask == True:
                masked_songs = self.df.loc[index, 'masked_songs']
                masked_tags = self.df.loc[index, 'masked_tags']
                masked_genres = self.df.loc[index, 'masked_genres']

                X[i, masked_songs] = 1
                X[i, masked_tags] = 1
                X[i, masked_genres] = 1

            else:
                X[i, songs] = 1
                X[i, tags] = 1
                X[i, genres] = 1

        if self.mode =='train':
            return X, y
        elif self.mode == 'predict':
            return X
        else:
            raise AttributeError('The mode parameters should be set to "train" or "predict"')

class TitleDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 df,
                 max_len,
                 dim_output,
                 batch_size=4,
                 mode='train',
                 shuffle=True):
        super(TitleDataGenerator, self).__init__()
        self.df = df
        self.max_len = max_len
        self.dim_output = dim_output
        self.batch_size =batch_size
        self.mode = mode
        self.shuffle = shuffle

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.array(self.df.index)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = np.zeros((self.batch_size, self.max_len), dtype=np.float32)
        y = np.zeros((self.batch_size, self.dim_output), dtype=np.float32)

        for i, index in enumerate(batch_indexes):
            songs = self.df.loc[index, 'songs']
            tags = self.df.loc[index, 'tags']
            title = self.df.loc[index, 'title']

            X[i,:] = title
            y[i,songs] = 1
            y[i,tags] = 1

        if self.mode == 'train':
            return X, y
        elif self.mode == 'predict':
            return X
        else:
            raise AttributeError('The mode parameters should be set to "train" or "predict"')




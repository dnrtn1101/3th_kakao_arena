import pandas as pd
import tensorflow as tf

# custom modules
from utils import write_pickle, load_pickle
from DataGenerator import TitleDataGenerator
from models import textCNN

def weighted_loss(alpha):
    def loss(y_true, y_pred):
        L = -tf.reduce_sum(y_true*tf.math.log(y_pred+1e-7) + alpha*(1-y_true)*tf.math.log(1-y_pred + 1e-7), axis=1)
        return L
    return loss

def train(args):
    # Preparing the dataset
    full = pd.read_json(args.data_path)
    mapping = load_pickle(args.mapping_path)
    embedding = load_pickle(args.embedding_path)
    dim_output = mapping['num_songs'] + mapping['num_tags']

    # TitleDataGenerator
    train_gen = TitleDataGenerator(df = full[(full['dataset'] == 0) & (full['title'].apply(lambda x: sum(x) > 0))],
                                   max_len = 10,
                                   dim_output = dim_output,
                                   batch_size=args.batch_size)
    val_gen = TitleDataGenerator(df = full[(full['dataset'] == 1) & (full['title'].apply(lambda x: sum(x) > 0))],
                                 max_len=10,
                                 dim_output=dim_output,
                                 batch_size=args.batch_size)

    model = textCNN(embedding_matrix = embedding,
                    max_len=10,
                    output_size=dim_output)
    optimizer = tf.optimizers.Adam(learning_rate=args.lr)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'./model_checkpoints/{args.name}',
        save_weight_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    model.compile(optimizer=optimizer, loss=weighted_loss(args.alpha))

    model.fit(train_gen, validation_data=val_gen, callbacks=[checkpoint], epochs=args.epochs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./output/full(0725_4_2).json')
    parser.add_argument('--mapping_path', type=str, default='./output/mapping(0718_4_2).pickle')
    parser.add_argument('--embedding_path', type=str, default='./output/embedding.pickle')
    parser.add_argument('--name', type=str, default='titleCNN')
    parser.add_argument('--alpha', type=float, default=0.55)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
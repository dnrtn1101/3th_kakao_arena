import pandas as pd
import tensorflow as tf

# custom modules
import models
import DataGenerator
from utils import write_pickle, load_pickle

def weighted_loss(alpha):
    def loss(y_true, y_pred):
        L = -tf.reduce_sum(y_true*tf.math.log(y_pred+1e-7) + alpha*(1-y_true)*tf.math.log(1-y_pred + 1e-7), axis=1)
        return L
    return loss

def train(args):
    # preparing the dataset
    full = pd.read_json(args.data_path)
    mapping = load_pickle(args.mapping_path)
    print(mapping['num_songs'], mapping['num_tags'], mapping['num_genres'])
    dim_input = mapping['num_songs'] + mapping['num_tags'] + mapping['num_genres'] + 2

    # DataGenerator
    train_gen = DataGenerator.DataGenerator(df = full.loc[full['dataset'] == 0],
                                            dim=dim_input,
                                            song_to_genres=mapping['song_to_genres'],
                                            mask=True,
                                            batch_size = args.batch_size)
    val_gen = DataGenerator.DataGenerator(df = full.loc[full['dataset'] == 1],
                                          dim=dim_input,
                                          song_to_genres=mapping['song_to_genres'],
                                          shuffle=False,
                                          batch_size=args.batch_size)

    # creating the model, the loss function, and the optimizer
    autoencoder = models.AutoEncoder(dim_input=dim_input, dim_latent=args.dim_latent)
    # optimizer = tf.optimizers.Adam(learning_rate=args.lr)
    optimizer = tf.optimizers.Adam(learning_rate=args.lr)
    checkpoint=tf.keras.callbacks.ModelCheckpoint(
        filepath=f'./model_checkpoints/{args.name}',
        save_weight_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    # compile and train the model.
    autoencoder.compile(optimizer=optimizer, loss=weighted_loss(args.alpha))

    history = autoencoder.fit(train_gen,
                              validation_data=val_gen,
                              callbacks=[checkpoint],
                              epochs=args.epochs)

    history.history['name'] = args.name
    write_pickle(fpath=f'./output/history_{args.name}.pickle',
                 obj=history.history)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./output/full(0718_5_3).json')
    parser.add_argument("--mapping_path", type=str, default='./output/mapping(0718_5_3).pickle')
    parser.add_argument("--name", type=str, default='dae_on0718')
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--dim_latent", type=int, default=256)
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
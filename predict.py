from utils import write_json, load_pickle
import numpy as np
import pandas as pd

import models
import DataGenerator

def generate_prediction(scores, topk, seen, start_from, mapping):
    candidates = np.argsort(-1 * scores) + start_from
    candidates = candidates.tolist()

    for i in seen:
        try:
            candidates.remove(i)
        except:
            pass

    candidates = candidates[:topk]
    candidates_id = list(map(lambda x: mapping[x], candidates))

    return candidates_id

def get_test_data(df, data):
    if data == 'train':
        return df[df['dataset'] == 0]
    elif data == 'my_val':
        return df[df['dataset'] == 1]
    elif data == 'val':
        return df[df['dataset'] == 2]
    elif data == 'test':
        return df[df['dataset'] == 3]
    else:
        raise AttributeError('The data parameters should be set to one of ["train", "my_val", "val", "test"]')

def predict(args):
    # Load the dataset to be predicted
    full = pd.read_json(args.data_path)
    test = get_test_data(full, args.data)
    test = test.reset_index(drop=True)

    # Load the item-id mapping
    mapping = load_pickle(args.mapping_path)
    dim_input = mapping['num_songs'] + mapping['num_tags'] + mapping['num_genres'] +2

    # Load the autoencoder model
    autoencoder = models.AutoEncoder(dim_input=dim_input, dim_latent=256)
    autoencoder.load_weights(args.checkpoint_path)

    # Building the test generator
    test_generator = DataGenerator.DataGenerator(test,
                                                 dim=dim_input,
                                                 song_to_genres=mapping['song_to_genres'],
                                                 batch_size=1,
                                                 shuffle = False,
                                                 mode= 'predict').__iter__()

    # Generating predictions
    predicts = []
    print("")
    for i, batches in enumerate(test_generator):
        print(f"{i}/{len(test)}", end='\r')
        output = autoencoder.predict(batches)[0]

        predicts.append({
            'id': test.loc[i, 'id'],
            'songs': generate_prediction(scores=output[:mapping['num_songs']],
                                         topk=100,
                                         seen=test.loc[i, 'songs'],
                                         start_from=0,
                                         mapping=mapping['inverse_mapping']),
            'tags': generate_prediction(scores=output[mapping['num_songs']:mapping['num_songs'] + mapping['num_tags']],
                                        topk=10,
                                        seen=test.loc[i, 'tags'],
                                        start_from=mapping['num_songs'],
                                        mapping=mapping['inverse_mapping'])
        })

    return predicts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./output/full(0718_5_3).json')
    parser.add_argument('--mapping_path', type=str, default='./output/mapping(0718_5_3).pickle')
    parser.add_argument('--checkpoint_path', type=str, default='./model_checkpoints/dae_on0718')
    parser.add_argument('--save_path', type=str, default= './results/results_on0718.json')
    parser.add_argument('--data', type=str, default='val')
    args = parser.parse_args()

    result_path = args.save_path
    write_json(predict(args), result_path)

if __name__ == '__main__':
    main()
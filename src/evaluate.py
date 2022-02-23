import sys
sys.path.append('./src')

import fire
import numpy as np
from utils import load_json

class Evaluator:
    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]

    def _ndcg(self, y_true, y_pred):
        dcg = 0.0
        for i,r in enumerate(y_pred):
            if r in y_true:
                dcg += 1.0 / np.log(i+2)
        return dcg / self._idcgs[len(y_true)]

    def _eval(self, y_true_fname, y_pred_fname):
        true_playlists = load_json(y_true_fname)
        true_dict = {g['id']: g for g in true_playlists}

        pred_playlists = load_json(y_pred_fname)

        true_indices = set([g['id'] for g in true_playlists])
        pred_indices = set([r['id'] for r in pred_playlists])

        if true_indices != pred_indices:
            raise Exception("결과의 플레이리스트 수가 올바르지 않습니다.")

        pred_song_counts = [len(p['songs']) for p in pred_playlists]
        pred_tag_counts = [len(p['tags']) for p in pred_playlists]

        if set(pred_song_counts) != set([100]):
            raise Exception("추천 곡 결과의 개수가 맞지 않습니다.")

        if set(pred_tag_counts) != set([10]):
            raise Exception("추천 태그 결과의 개수가 맞지 않습니다.")

        pred_unique_song_counts = [len(set(p['songs'])) for p in pred_playlists]
        pred_unique_tag_counts = [len(set(p['tags'])) for p in pred_playlists]

        if set(pred_unique_song_counts) != set([100]):
            raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")

        if set(pred_unique_tag_counts) != set([10]):
            raise Exception("한 플레이리스트에 중복된 태그 추천은 허용되지 않습니다.")

        music_ndcg = 0.0
        tag_ndcg = 0.0

        for y_pred in pred_playlists:
            y_true = true_dict[y_pred['id']]
            music_ndcg += self._ndcg(y_true['songs'], y_pred['songs'][:100])
            tag_ndcg += self._ndcg(y_true['tags'], y_pred['tags'][:10])

        music_ndcg = music_ndcg / len(pred_playlists)
        tag_ndcg = tag_ndcg / len(pred_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def evaluate(self, y_true_fname, y_pred_fname):
        try:
            music_ndcg, tag_ndcg, score = self._eval(y_true_fname, y_pred_fname)
            print(f"Music nDCG: {music_ndcg:.6}")
            print(f"Tag nDCG: {tag_ndcg:.6}")
            print(f"Score: {score:.6}")
        except Exception as e:
            print(e)

if __name__ == '__main__':
    fire.Fire(Evaluator)
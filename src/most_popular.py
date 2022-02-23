import sys
sys.path.append('./src')

import fire
from tqdm import tqdm
from utils import load_json, write_json, remove_seen, most_popular

class MostPopular:
    def _generate_answers(self, train, questions):
        _, song_mp = most_popular(train, 'songs', 200)
        _, tag_mp = most_popular(train, 'tags', 100)

        answers = []

        for q in tqdm(questions):
            answers.append({
                'id':q['id'],
                'songs':remove_seen(q['songs'], song_mp)[:100], # 현재 수록된 곡(q['songs'])에 없는 인기곡만 선택
                'tags':remove_seen(q['tags'], tag_mp)[:10],
            })

        return answers

    def run(self, train_fname, question_fname):
        print("Loading train file...")
        train = load_json(train_fname)

        print("Loading question file...")
        questions = load_json(question_fname)

        print("Writing answers...")
        answers = self._generate_answers(train, questions)
        write_json(answers, 'results/result(most_popular).json')

if __name__ == "__main__":
    fire.Fire(MostPopular)
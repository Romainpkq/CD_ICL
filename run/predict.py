import json
import os
import math


def predict(origin_prob_path, reverse_prob_path, test_path, label_names):
    with open(origin_prob_path, 'r', encoding='utf-8') as f1:
       origin_data_list = list(f1)

    with open(reverse_prob_path, 'r', encoding='utf-8') as f2:
       reverse_data_list = list(f2)

    labels = list(label_names.keys())

    print(label_names)

    origin_predict_labels = []
    reverse_predict_labels = []
    cd_reverse_origin_predict_labels = []

    for i in range(len(origin_data_list)):
        origin_data = json.loads(origin_data_list[i])
        reverse_data = json.loads(reverse_data_list[i])

        origin_probs = [math.exp(origin_data[label]) for label in labels]
        reverse_probs = [math.exp(reverse_data[label]) for label in labels]

        #print(origin_probs)

        origin_probs = [prob / sum(origin_probs) for prob in origin_probs]
        reverse_probs = [prob / sum(reverse_probs) for prob in reverse_probs]

        cd_origin_probs = [origin_probs[i] * ((origin_probs[i] / reverse_probs[i]) ** 1.0) for i in range(len(origin_probs))]

        origin_predict_labels.append(labels[origin_probs.index(max(origin_probs))])
        reverse_predict_labels.append(labels[reverse_probs.index(max(reverse_probs))])

        cd_reverse_origin_predict_labels.append(labels[cd_origin_probs.index(max(cd_origin_probs))])

    #print(origin_predict_labels)
    golden_labels = []

    with open(test_path, 'r', encoding='utf-8') as f2:
        test_data_list = list(f2)

        for i in range(len(test_data_list)):
            test_data = json.loads(test_data_list[i])

            if 'trec' in test_path:
                golden_labels.append(test_data['label_coarse'])
            else:
                golden_labels.append(test_data['label'])

    origin_num = 0
    reverse_num = 0
    cd_origin_num = 0

    for i in range(len(golden_labels)):
        if label_names[origin_predict_labels[i]] == golden_labels[i]:
            origin_num += 1

        if label_names[reverse_predict_labels[i]] == golden_labels[i]:
            reverse_num += 1

        if label_names[cd_reverse_origin_predict_labels[i]] == golden_labels[i]:
            cd_origin_num += 1

    print('origin: {}\nCD origin: {}'.format(origin_num / len(golden_labels), cd_origin_num / len(golden_labels)))


subj_tp_dict = {
    0: "</E>Input: </text>\nType: objective",
    1: "</E>Input: </text>\nType: subjective"
    }

sst2_tp_dict = {
    0: "</E>Review: </text>\nSentiment: negative",
    1: "</E>Review: </text>\nSentiment: positive"
    }

sst5_tp_dict = {
    0: "</E>Review: </text>\nSentiment: terrible",
    1: "</E>Review: </text>\nSentiment: bad",
    2: "</E>Review: </text>\nSentiment: okay",
    3: "</E>Review: </text>\nSentiment: good",
    4: "</E>Review: </text>\nSentiment: great",
}

cr_tp_dict = {
    0: "</E>Review: </text>\nSentiment: negative",
    1: "</E>Review: </text>\nSentiment: positive"
}

ag_news_tp_dict = {
    0: "</E>Input: </text>\nType: world",
    1: "</E>Input: </text>\nType: sports",
    2: "</E>Input: </text>\nType: business",
    3: "</E>Input: </text>\nType: technology",
}

mnli_tp_dict = {
        0: "</E>Premise: </text1>\nHypothesis: </text>\nPrediction: entail",
        1: "</E>Premise: </text1>\nHypothesis: </text>\nPrediction: neutral",
        2: "</E>Premise: </text1>\nHypothesis: </text>\nPrediction: contradiction"
        }

qnli_tp_dict = {
        0: "</E></text1> Can we know </text>? Yes",
        1: "</E></text1> Can we know </text>? No"
        }

trec_tp_dict = {
    0: "</E>Question: </text>\nType: description",
    1: "</E>Question: </text>\nType: entity",
    2: "</E>Question: </text>\nType: expression",
    3: "</E>Question: </text>\nType: human",
    4: "</E>Question: </text>\nType: location",
    5: "</E>Question: </text>\nType: number",
}

dbpedia_tp_dict = {
        0: "</E>Content:  </text>\nTopic: company",
        1: "</E>Content:  </text>\nTopic: school",
        2: "</E>Content:  </text>\nTopic: artist",
        3: "</E>Content:  </text>\nTopic: athlete",
        4: "</E>Content:  </text>\nTopic: politics",
        5: "</E>Content:  </text>\nTopic: transportation",
        6: "</E>Content:  </text>\nTopic: building",
        7: "</E>Content:  </text>\nTopic: nature",
        8: "</E>Content:  </text>\nTopic: village",
        9: "</E>Content:  </text>\nTopic: animal",
        10: "</E>Content:  </text>\nTopic: plant",
        11: "</E>Content:  </text>\nTopic: album",
        12: "</E>Content:  </text>\nTopic: film",
        13: "</E>Content:  </text>\nTopic: book"
        }

task_dicts = {
        'sst2': sst2_tp_dict,
        'subj': subj_tp_dict,
        "sst5": sst5_tp_dict,
        'cr': cr_tp_dict,
        "ag_news": ag_news_tp_dict,
        'mnli': mnli_tp_dict,
        'qnli': qnli_tp_dict,
        'trec': trec_tp_dict,
        'dbpedia': dbpedia_tp_dict
        }

test_split={
            'sst2': 'test',
            "subj": 'test',
            "sst5": 'test',
            "cr": 'test',
            "ag_news": 'test',
            'mnli': 'validation', # cannot get gold labels for the test split
            "qnli": 'validation',
            'dbpedia': 'test',
            'trec': 'test'
    }

if __name__ == "__main__":
    task_names = ['sst2', 'cr', 'sst5', 'subj', 'qnli', 'mnli', 'ag_news']

    model_name = 'Meta-Llama-3.1-8B'
    model_name1 = 'Meta-Llama-3.1-8B'

    seeds = [1, 66, 250]
    methods = ['random', 'bm25', 'topk']

    prob_dir = './results'
    prob_dir1 = './results'

    origin_prob_path1 = os.path.join(prob_dir, model_name)
    origin_prob_path11 = os.path.join(prob_dir1, model_name1)

    test_dir = ''

    for task_name in task_names:
        print('task_name: ', task_name)
        origin_prob_path2 = os.path.join(origin_prob_path1, task_name)
        origin_prob_path12 = os.path.join(origin_prob_path11, task_name)

        for method in methods:
            for seed in seeds:
                print('')
                print('method: {} seed: {}'.format(method, seed))
                origin_name = 'no_instruct_origin_' + method + '_16_shots_1_seed_' + str(seed) + '_seed1_prob.jsonl'
                reverse_name = 'no_instruct_reverse_mapping_seed_' + method + '_16_shots_1_seed_' + str(seed) + '_seed1_prob.jsonl'

                origin_prob_path = os.path.join(origin_prob_path2, origin_name)
                reverse_prob_path = os.path.join(origin_prob_path12, reverse_name)

                test_path1 = os.path.join(test_dir, task_name)
                test_path = os.path.join(test_path1, test_split[task_name] + '.jsonl')

                label_names = dict()

                for key, value in task_dicts[task_name].items():
                    label = value.split()[-1]
                    label_names[label] = key

                predict(origin_prob_path, reverse_prob_path, test_path, label_names)


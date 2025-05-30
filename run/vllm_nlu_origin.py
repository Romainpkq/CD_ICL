import json
import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
import torch
import random
import os
import numpy as np
from vllm import LLM, SamplingParams

# openicl
from openicl import DatasetReader, PromptTemplate, ZeroRetriever, RandomRetriever, BM25Retriever, TopkRetriever, GenInferencer
from datasets import load_dataset
from accelerate import Accelerator
from openicl import CDRandomRetriever, CDBM25Retriever, CDTopkRetriever


def retrival_text(data, template, ice_num, batch_size, ret_method, sentence_model_name='', tokenizer_name='', seed=1, prompt_type='origin', seed1=1):
    if ice_num == 0:
        retriever = ZeroRetriever(data, index_split='train', test_split='test')
    elif ret_method == 'random':
        retriever = CDRandomRetriever(data, seed=seed, ice_num=ice_num, test_split='test')
    elif ret_method == 'bm25':
        retriever = CDBM25Retriever(data, ice_num=ice_num, index_split='train', test_split='test')
    elif ret_method == 'topk':
        retriever = CDTopkRetriever(data, ice_num=ice_num, sentence_transformers_model_name=sentence_model_name, tokenizer_name=sentence_model_name, batch_size=batch_size)

    ice_idx_list = retriever.retrieve()

    prompt_list = []

    labels = retriever.get_labels(template)
    print(labels)

    for idx, ice_idx in enumerate(ice_idx_list):
        if prompt_type == 'origin':
            ice = retriever.generate_ice(ice_idx, ice_template=template)
        elif prompt_type == 'reverse_mapping_seed':
            ice = retriever.generate_reverse_mapping_seed_ice(ice_idx, ice_template=template, seed=seed1)

        for label in labels:
            prompt = retriever.generate_label_prompt(idx, ice, label, template)
            prompt_list.append(prompt)

    print(len(prompt_list))

    return prompt_list, len(labels)


subj_tp_dict = {
    0: "</E>Input: </text>\nType: objective",
    1: "</E>Input: </text>\nType: subjective"
    }

subj_template = PromptTemplate(subj_tp_dict, {'text': '</text>'}, ice_token='</E>')

sst2_tp_dict = {
    0: "</E>Review: </text>\nSentiment: negative",
    1: "</E>Review: </text>\nSentiment: positive"
    }

sst2_template = PromptTemplate(sst2_tp_dict, {'text': '</text>'}, ice_token='</E>')

sst5_tp_dict = {
    0: "</E>Review: </text>\nSentiment: terrible",
    1: "</E>Review: </text>\nSentiment: bad",
    2: "</E>Review: </text>\nSentiment: okay",
    3: "</E>Review: </text>\nSentiment: good",
    4: "</E>Review: </text>\nSentiment: great",
}

sst5_template = PromptTemplate(sst5_tp_dict, {'text': '</text>'}, ice_token='</E>')

cr_tp_dict = {
    0: "</E>Review: </text>\nSentiment: negative",
    1: "</E>Review: </text>\nSentiment: positive"
}

cr_template = PromptTemplate(cr_tp_dict, {'text': '</text>'}, ice_token='</E>')

ag_news_tp_dict = {
    0: "</E>Input: </text>\nType: world",
    1: "</E>Input: </text>\nType: sports",
    2: "</E>Input: </text>\nType: business",
    3: "</E>Input: </text>\nType: technology",
}

ag_news_template = PromptTemplate(ag_news_tp_dict, {'text': '</text>'}, ice_token='</E>')

trec_tp_dict = {
    0: "</E>Question: </text>\nType: description",
    1: "</E>Question: </text>\nType: entity",
    2: "</E>Question: </text>\nType: expression",
    3: "</E>Question: </text>\nType: human",
    4: "</E>Question: </text>\nType: location",
    5: "</E>Question: </text>\nType: number",
}

trec_template = PromptTemplate(trec_tp_dict, {'text': '</text>'}, ice_token='</E>')

mnli_tp_dict = {
        0: "</E>Premise: </text1>\nHypothesis: </text>\nPrediction: entailment",
        1: "</E>Premise: </text1>\nHypothesis: </text>\nPrediction: neutral",
        2: "</E>Premise: </text1>\nHypothesis: </text>\nPrediction: contradiction"
        }

mnli_template = PromptTemplate(mnli_tp_dict, {'text1': '</text1>', 'text2': '</text>'}, ice_token='</E>')

qnli_tp_dict = {
        0: "</E></text1> Can we know </text>? Yes.",
        1: "</E></text1> Can we know </text>? No."
        }
qnli_template = PromptTemplate(qnli_tp_dict, {'text1': '</text1>', 'text2': '</text>'}, ice_token='</E>')


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

dbpedia_template = PromptTemplate(dbpedia_tp_dict, {'content': '</text>'}, ice_token='</E>')

templates = {
        'sst2': sst2_template,
        'subj': subj_template,
        "sst5": sst5_template,
        'cr': cr_template,
        "ag_news": ag_news_template,
        'mnli': mnli_template,
        'qnli': qnli_template,
        'dbpedia': dbpedia_template,
        'trec': trec_template
        }

task_dicts = {
        'sst2': sst2_tp_dict,
        'subj': subj_tp_dict,
        "sst5": sst5_tp_dict,
        'cr': cr_tp_dict,
        "ag_news": ag_news_tp_dict,
        'mnli': mnli_tp_dict,
        'qnli': qnli_tp_dict,
        'dbpedia': dbpedia_tp_dict,
        'trec': trec_tp_dict
        }

input_columns={'sst2': ["text"],
            'subj': ['text'],
            "sst5": ["text"],
            "cr": ["text"],
            "ag_news": ["text"],
            "trec": ["text"],
            'mnli': ['text1', 'text2'],
            "qnli": ["text1", "text2"],
            'dbpedia': ['content']
            }

output_columns={'sst2': 'label',
             'subj': 'label',
             "sst5": 'label',
             'cr': 'label',
             "ag_news": 'label',
             "trec": 'label_coarse',
             'mnli': 'label',
             "qnli": 'label',
             'dbpedia': 'label'
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, default="Meta-Llama-3.1-8B", required=False, help='model name in the hub or local path')
    parser.add_argument('--sentence-model-name-or-path', type=str, default="all-mpnet-base-v2", required=False, help='sentence model name in the hub or local path')
    parser.add_argument('--train-dir', type=str, default=None, required=False, help='train dir')
    parser.add_argument('--test-dir', type=str, default=None, required=False, help='test dir')
    parser.add_argument('--output-dir','-o', type=str, default="./results", required=False, help='save dir')
    parser.add_argument('--search-algorithm', '-sa', type=str, default='beam', help='search algorithms: sample, beam')
    parser.add_argument('--output-name','-on', type=str, default="", required=False, help='output name correct or wrong')
    parser.add_argument('--template', '-tp', type=str, default="", help='the template')
    parser.add_argument('--ret-method', '-ret', type=str, default="random", help='retrieval method')
    parser.add_argument('--n-shot', '-n', type=str, default='0', help='inference with n demonstrative samples')
    parser.add_argument('--ins', type=bool, default=False, help='use instruction or not')
    parser.add_argument('--temperature', '-t', type=float, default=0.1, help='temperature: 0.7 for text generation')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--seed', type=str, default='1', help='seed')
    parser.add_argument('--seed1', type=str, default='1', help='seed1 for reverse')
    parser.add_argument('--task-names', '-tn', type=str, default='sst2', help='the name of the task')
    parser.add_argument('--prompt-type', '-pt', type=str, default='origin', help='the types of prompt')
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    sentence_model_name_or_path = args.sentence_model_name_or_path
    model_name = model_name_or_path.split('/')[-1]

    train_dir = args.train_dir
    test_dir = args.test_dir
    output_dir = args.output_dir
    output_name = args.output_name

    search = args.search_algorithm

    n_shot = args.n_shot
    ret_method = args.ret_method

    temperature = args.temperature
    batch_size = args.batch_size
    ins = args.ins
    seed = args.seed
    seed1 = args.seed1

    task_names = args.task_names
    prompt_type = args.prompt_type

    # load model
    model = LLM(model_name_or_path, tensor_parallel_size=1, gpu_memory_utilization=0.8, swap_space=1, enforce_eager=True)
    to_use_fast = False
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=to_use_fast)

    sampling_param = SamplingParams(
            top_p=1, temperature=0, max_tokens=1,
            use_beam_search=False, best_of=1, stop_token_ids=[tokenizer.eos_token_id],
            prompt_logprobs=1
            )

    output_path1 = os.path.join(output_dir, model_name)

    task_names = task_names.split(',')

    for task_name in task_names:
        # load dataset
        train_path1 = os.path.join(train_dir, task_name)
        test_path1 = os.path.join(test_dir, task_name)

        train_path = os.path.join(train_path1, 'train.jsonl')
        test_path = os.path.join(test_path1, test_split[task_name] + '.jsonl')

        dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
        data = DatasetReader(dataset, input_columns=input_columns[task_name], output_column=output_columns[task_name])

        print(data)
        output_path2 = os.path.join(output_path1, task_name)

        print(input_columns[task_name])
        #test_data = data['test'][input_columns[task_name]]
        os.makedirs(output_path2, exist_ok=True)

        # get the template
        template = templates[task_name]

        # get the label name
        key_word = template.template[0].split('</text>')[-1].split()[0]
        test_pos = template.template[0].rindex(key_word) + len(key_word)

        all_labels = [template.template[i][test_pos:] for i in range(len(template.template))]
        all_labels_lens = [len(tokenizer.tokenize(label)) for label in all_labels]

        n_shots = n_shot.split(',')

        for shots in n_shots:
            shots = int(shots)
            methods = ret_method.split(',')
            
            for method in methods:
                # instruction
                instruct_name = ''

                #for seed in seeds.split(','):
                seed = int(seed)
                seed1 = int(seed1)

                few_shots, labels_num = retrival_text(data, template, shots, batch_size, method, sentence_model_name=sentence_model_name_or_path, tokenizer_name=sentence_model_name_or_path, seed=seed, prompt_type=prompt_type, seed1=seed1)

                if shots == 0:
                    output_name1 = instruct_name + str(shots) + '_shots'
                else:
                    output_name1 = instruct_name + output_name + '_'  + prompt_type + '_' + method + '_' + str(shots) + '_shots_' + str(seed) + '_seed_' + str(seed1) + '_seed1'

                output_path = os.path.join(output_path2, output_name1)

                prompt = [example.strip() for example in few_shots]

                print('prompt:', prompt[0])
                print('label_num:', labels_num)
                # Generate
                torch.manual_seed(0)

                with open(output_path, 'w', encoding='utf-8') as fo,open(output_path + '_prob.jsonl', 'w', encoding='utf-8') as fo1,open(output_path+".hyp", 'w', encoding='utf-8') as fo2:
                    decoded_tokens = model.generate(prompt, sampling_params=sampling_param)
                    
                    results = []
                    prob_dict = dict()

                    for i in range(len(decoded_tokens)):
                        item = decoded_tokens[i]

                        pos = i % labels_num

                        all_logprobs = item.prompt_logprobs[-all_labels_lens[pos]]
                        print(item.outputs[0].text.split("\n")[0], file=fo2, flush=True)

                        if i % labels_num == 0 and i != 0:
                                results.append(prob_dict)
                                prob_dict = dict()

                        for key, value in all_logprobs.items():
                            new_value = value.decoded_token.strip()

                            if new_value not in prob_dict:
                                prob_dict[new_value] = value.logprob

                    results.append(prob_dict)

                    for result in results:
                        json.dump(result, fo1)
                        fo1.write('\n')

                    for i in range(len(prompt)):
                        if i % labels_num == 0:
                            sample = prompt[i]
                            i = i // labels_num
                            print('example: {}'.format(str(i)), file=fo, flush=True)
                            print(sample, file=fo, flush=True)

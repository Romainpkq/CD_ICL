model_dir1=/mnt/data/huggingface/models/Llama3.1
model_name1=Meta-Llama-3.1-8B
#model_name1=Meta-Llama-3.1-8B-Instruct

model_dir=/mnt/data/huggingface/models/llama3.2
model_name=Llama-3.2-1B
model_name1=Llama-3.2-3B
#model_name=Llama-3.2-1B-Instruct

task_name=sst2,cr,subj,sst5,qnli,mnli,ag_news
#task_name1=qnli,ag_news
#task_name=dbpedia,trec
#task_name=ag_news
task_name=sst2

prompt_type=origin
prompt_type1=reverse_mapping_seed

CUDA_VISIBLE_DEVICES=4 nohup python vllm_nlu_origin.py -tn ${task_name} -n 16 -pt ${prompt_type} --model-name-or-path ${model_dir}/${model_name} -on no_instruct --seed1 66 -ret random,bm25,topk > ./logs/8b_reverse_mapping_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python vllm_nlu_origin.py -tn ${task_name} -n 16 -pt ${prompt_type1} --model-name-or-path ${model_dir}/${model_name} -on no_instruct --seed1 66 -ret random,bm25,topk > ./logs/8b_reverse_mapping_2.log 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python vllm_nlu_origin.py -tn ${task_name} -n 16 -pt ${prompt_type} --model-name-or-path ${model_dir}/${model_name1} -on no_instruct --seed1 66 -ret random > ./logs/8b_reverse_mapping_3.log 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python vllm_nlu_origin.py -tn ${task_name} -n 16 -pt ${prompt_type1} --model-name-or-path ${model_dir}/${model_name1} -on no_instruct --seed1 66 -ret random > ./logs/8b_reverse_mapping_4.log 2>&1 &

#CUDA_VISIBLE_DEVICES=2 nohup python vllm_nlu_origin_sup.py -tn ${task_name} -n 16 -pt ${prompt_type} --model-name-or-path ${model_dir}/${model_name} -on no_instruct --seed1 250 -ret random,bm25,topk > ./logs/8b_reverse_mapping_3.log 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python vllm_nlu_origin_sup.py -tn ${task_name} -n 16 -pt ${prompt_type1} --model-name-or-path ${model_dir}/${model_name} -on no_instruct --seed1 250 -ret random,bm25,topk > ./logs/8b_reverse_mapping_4.log 2>&1 &

#CUDA_VISIBLE_DEVICES=2 nohup python vllm_nlu_origin.py -tn ${task_name} -n 16 -pt ${prompt_type} --model-name-or-path ${model_dir}/${model_name1} -on no_instruct -ret random,bm25,topk > ./logs/8b_reverse_mapping_3.log 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python vllm_nlu_origin.py -tn ${task_name} -n 16 -pt ${prompt_type1} --model-name-or-path ${model_dir}/${model_name1} -on no_instruct -ret random,bm25,topk > ./logs/8b_reverse_mapping_4.log 2>&1 &


"""Topk Retriever"""

from openicl import DatasetReader
from openicl.icl_dataset_reader import DatasetEncoder
from openicl.icl_retriever import BaseRetriever
from openicl.utils.collators import DataCollatorWithPaddingAndCuda
from openicl.utils.logging import get_logger
import torch
from typing import List, Union, Optional
from tqdm import trange
import numpy as np
from openicl import DatasetReader, PromptTemplate
from torch.utils.data import DataLoader
from typing import Optional
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import tqdm
import random
import faiss
import copy
import numpy as np
from accelerate import Accelerator

logger = get_logger(__name__)


class CDTopkRetriever(BaseRetriever):
    """Topk In-context Learning Retriever Class
        Class of Topk Retriever.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:`model`.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
    """
    model = None

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name: Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        gen_datalist = self.dataset_reader.generate_input_field_corpus(self.test_ds)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        self.encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        self.dataloader = DataLoader(self.encode_dataset, batch_size=self.batch_size, collate_fn=co)

        self.model = SentenceTransformer(sentence_transformers_model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.index = self.create_index()

        self.labels = list(sorted(list(set(self.index_ds[self.dataset_reader.output_column]))))
        self.labels_index = {label: [] for label in self.labels}

        for i in range(len(self.index_ds[self.dataset_reader.output_column])):
            self.labels_index[self.index_ds[self.dataset_reader.output_column][i]].append(i)

    def create_index(self):
        self.select_datalist = self.dataset_reader.generate_input_field_corpus(self.index_ds)
        encode_datalist = DatasetEncoder(self.select_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        dataloader = DataLoader(encode_datalist, batch_size=self.batch_size, collate_fn=co)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension()))
        res_list = self.forward(dataloader, process_bar=True, information="Creating index for index set...")
        id_list = np.array([res['metadata']['id'] for res in res_list])
        self.embed_list = np.stack([res['embed'] for res in res_list])
        index.add_with_ids(self.embed_list, id_list)
        return index

    def knn_search(self, ice_num):
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")
        rtr_idx_list = [[] for _ in range(len(res_list))]
        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(embed, ice_num)[1][0].tolist()
            rtr_idx_list[idx] = near_ids
        return rtr_idx_list

    def forward(self, dataloader, process_bar=False, information=''):
        res_list = []
        _dataloader = copy.deepcopy(dataloader)
        if process_bar:
            logger.info(information)
            _dataloader = tqdm.tqdm(_dataloader, disable=not self.is_main_process)
        for _, entry in enumerate(_dataloader):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                raw_text = self.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True, verbose=False)
                res = self.model.encode(raw_text, show_progress_bar=False)
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def retrieve(self):
        return self.knn_search(self.ice_num)

    def generate_reverse_mapping_seed_ice(self, idx_list: List[int], ice_template: Optional[PromptTemplate] = None, seed: int = 1) -> str:
        generated_ice_list = []
        dr = self.dataset_reader

        demo_text = [self.index_ds[idx] for idx in idx_list]
        #print(demo_text)
        demo_label = [self.index_ds[idx][dr.output_column] for idx in idx_list]

        # get the demo and the idx of the demo
        demo_label_idxs = dict()

        for idx in range(len(demo_label)):
            if demo_label[idx] not in demo_label_idxs:
                demo_label_idxs[demo_label[idx]] = [idx]
            else:
                demo_label_idxs[demo_label[idx]].append(idx)

        # get the random text for each label
        random_demo_text = dict()
        num_label = len(self.labels)

        offsets = list(range(1, num_label))

        random.seed(seed)

        for key, value in self.labels_index.items():
            random_demo_text[key] = [self.index_ds[idx] for idx in random.sample(value, len(demo_label))]

        new_demo_text = []

        for label in demo_label:
            key = random.sample(offsets, 1)[0]
            new_demo_text.append(random_demo_text[(label + key) % num_label].pop())

        #print(new_demo_text)
        generated_ice_list = [ice_template.generate_ice_item(new_demo_text[idx], demo_label[idx]) for idx in range(len(demo_label))]
        generated_ice = self.ice_separator.join(generated_ice_list) + self.ice_eos_token
        return generated_ice

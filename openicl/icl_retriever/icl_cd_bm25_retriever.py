"""BM25 Retriever"""

from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger
from typing import List, Union, Optional
from rank_bm25 import BM25Okapi
import numpy as np
import random
from openicl import DatasetReader, PromptTemplate
from tqdm import trange
from accelerate import Accelerator
from nltk.tokenize import word_tokenize

logger = get_logger(__name__)


class CDBM25Retriever(BaseRetriever):
    """BM25 In-context Learning Retriever Class
        Class of BM25 Retriever.
        
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
        index_corpus (:obj:`List[str]`) : A corpus created from the input field data of :obj:`index_ds`.
        test_corpus (:obj:`List[str]`) : A corpus created from the input field data of :obj:`test_ds`.
        bm25 (:obj:`BM250kapi`): An instance of :obj:`BM250kapi` class, initialized using :obj:`index_ds`.
    """
    bm25 = None
    index_corpus = None
    test_corpus = None

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 accelerator: Optional[Accelerator] = None
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.index_corpus = [word_tokenize(data) for data in
                             self.dataset_reader.generate_input_field_corpus(self.index_ds)]
        self.bm25 = BM25Okapi(self.index_corpus)
        self.test_corpus = [word_tokenize(data) for data in
                            self.dataset_reader.generate_input_field_corpus(self.test_ds)]

        self.labels = list(sorted(list(set(self.index_ds[self.dataset_reader.output_column]))))
        self.labels_index = {label: [] for label in self.labels}

        for i in range(len(self.index_ds[self.dataset_reader.output_column])):
            self.labels_index[self.index_ds[self.dataset_reader.output_column][i]].append(i)

    def retrieve(self) -> List[List]:
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
            query = self.test_corpus[idx]
            scores = self.bm25.get_scores(query)
            near_ids = list(np.argsort(scores)[::-1][:self.ice_num])
            near_ids = [int(a) for a in near_ids]
            rtr_idx_list.append(near_ids)
        return rtr_idx_list

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

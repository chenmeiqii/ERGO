import os
import torch
import pickle
import numpy as np
from lxml import etree
import collections
import logging
from torch.utils.data import Dataset
import math
class ESL_examples:
    def __init__(self, topic_id, doc_id, doc, sentences, all_token, events, causal_links):
        self.topic_id = topic_id
        self.doc_id = doc_id
        self.doc = doc
        self.sentences = sentences
        self.events = events
        self.causal_links = causal_links
        self.all_token = all_token

class ESL_features(object):
    def __init__(self, topic_id, doc_id,
                 enc_text, enc_tokens, sentences,
                 enc_input_ids, enc_mask_ids, global_attention_mask,
                 t1_pos, t2_pos, mh_self_attention_mask,
                 target, rel_type, event_pairs
                 ):
        self.topic_id = topic_id
        self.doc_id = doc_id
        self.enc_text = enc_text
        self.enc_tokens = enc_tokens
        self.sentences = sentences
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        self.global_attention_mask = global_attention_mask
        self.t1_pos = t1_pos
        self.t2_pos = t2_pos
        self.mh_self_attention_mask = mh_self_attention_mask
        self.target = target
        self.rel_type = rel_type
        self.event_pairs = event_pairs

class ESL_dataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def collate_fn(self, batch):
        enc_input_ids = torch.tensor([f.enc_input_ids for f in batch])
        enc_mask_ids = torch.tensor([f.enc_mask_ids for f in batch])
        global_attention_mask = torch.tensor([f.global_attention_mask for f in batch])
        t1_pos = [f.t1_pos for f in batch]
        t2_pos = [f.t2_pos for f in batch]
        mh_self_attention_mask = torch.tensor([f.mh_self_attention_mask for f in batch])
        target = [f.target for f in batch]
        rel_type = [f.rel_type for f in batch]
        return (enc_input_ids, enc_mask_ids, global_attention_mask, t1_pos, t2_pos, mh_self_attention_mask, target, rel_type)


class ESL_processor(object):
    def __init__(self, args, tokenizer, logger):
        self.args = args
        self.tokenizer = tokenizer
        self.logger = logger
    def convert_features_to_dataset(self, features):
        dataset = ESL_dataset(features)
        return dataset

    def load_and_cache_examples(self, cache_path):
        examples = pickle.load(open(cache_path, 'rb'))
        print(f"load examples from {cache_path}")
        return examples

    def load_and_cache_features(self, cache_path):
        features = pickle.load(open(cache_path, 'rb'))
        print(f"load features from {cache_path}")
        return features

    def generate_dataloader(self, set_type):
        assert (set_type in ['train', 'dev'])
        cache_event_path = os.path.join(self.args.cache_path,
                                        "{}_{}_{}_{}_events.pkl".format(self.args.dataset_type, self.args.model_type, self.args.inter_or_intra,
                                                                     set_type))
        cache_feature_path = os.path.join(self.args.cache_path,
                                          "{}_{}_{}_{}_features.pkl".format(self.args.dataset_type, self.args.model_type, self.args.inter_or_intra,
                                                                         set_type))

        examples = self.load_and_cache_examples(cache_event_path)

        features = self.load_and_cache_features(cache_feature_path)

        dataset = self.convert_features_to_dataset(features)

        return examples, features, dataset

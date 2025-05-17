#2-step : loads the data, splits it into train/validation


import torch
import numpy as np
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import copy
from collections import defaultdict
import torch.nn.functional as F

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

import numpy as np
import random
import scipy.sparse as sp
import torch.utils.data as data
import torch
from os.path import join as path_join


# def load_text_weight(task, domain, device):
#     weight = np.load(path_join('../pretrained_parameters', '{}_{}_item_jina.npy'.format(task, domain)))
#     return torch.from_numpy(weight).to(device)

def load_text_weight(task, domain, device):
    # custom path for your project structure
    base_path = f'/home/n.dholakia002/LLMCDSR/LLMCDSR/generation/tuned_generation/embeddings/{task}'
    file_map = {'A': 'A_item_jina.npy', 'B': 'B_item_jina.npy'}
    weight = np.load(f'{base_path}/{file_map[domain]}')
    return torch.from_numpy(weight).to(device)



class SeqPTDataset:
    def __init__(self, args, domain='B'):
        import os
        import pickle

        with open(f'../data/{args.dataset}/num.txt', 'r') as f:
            Anum = int(f.readline().strip())
            Bnum = int(f.readline().strip())

        # ✅ Load item_set mapping (asin → index)
        if domain == 'A':
            with open(f'../data/{args.dataset}/item_set_A.pkl', 'rb') as f:
                item_set = pickle.load(f)
        elif domain == 'B':
            with open(f'../data/{args.dataset}/item_set_B.pkl', 'rb') as f:
                item_set = pickle.load(f)

        # ✅ Reverse mapping: index → asin-like IDs (used in parsed-items)
        index2id = {v: k for k, v in item_set.items()}

        # ✅ Load allowed_items (from parsed-items filenames)
        parsed_item_dir = f'/home/n.dholakia002/LLMCDSR/LLMCDSR/generation/tuned_generation/parsed-items/{args.dataset}'
        allowed_items = {
        index2id[int(f.split('_')[1].replace('.txt', ''))]
        for f in os.listdir(parsed_item_dir)
        if f.endswith('.txt') and int(f.split('_')[1].replace('.txt', '')) in index2id
        }

        print(f"[DEBUG] Number of allowed_items in parsed-items: {len(allowed_items)}")

        usernum = 0
        user_train, user_valid = {}, {}

        # ✅ Process domain-specific train_X.txt
        with open(f'../data/{args.dataset}/train_{domain}.txt', 'r') as f:
            for line in f:
                u, is_ = line.rstrip().split('\t')
                u = hash(u) % (10**8)

                # ✅ Map item indices back to original IDs
                is_ = [index2id[int(i)] for i in is_.split(',') if int(i) in index2id]

                if len(is_) < 2:
                    continue  # ⛔ skip users with <2 items

                # ✅ Filter by allowed_items (parsed-items)
                filtered_items = [i for i in is_ if i in allowed_items]

                if len(filtered_items) < 2:
                    print(f"[DEBUG] User {u} original items: {is_}")
                    print(f"[DEBUG] Allowed items sample: {list(allowed_items)[:10]}")
                    intersection = set(is_).intersection(allowed_items)
                    print(f"[DEBUG] Intersection with allowed_items: {intersection}")
                    print(f"[DEBUG] Skipping user {u}: only {len(filtered_items)} items after filtering: {filtered_items}")
                    continue

                user_train[u], user_valid[u] = self._train_valid_split(filtered_items)
                usernum = max(u, usernum)

        # ✅ Process overlapped users
        current_users = usernum
        with open(f'../data/{args.dataset}/train_overlap.txt', 'r') as f:
            for line in f:
                u, is_ = line.rstrip().split('\t')
                u = (hash(u) % (10**8)) + current_users

                is_ = [index2id[int(i)] for i in is_.split(',') if int(i) in index2id]

                if len(is_) < 2:
                    continue

                filtered_items = [i for i in is_ if i in allowed_items]

                if len(filtered_items) < 2:
                    continue

                user_train[u], user_valid[u] = self._train_valid_split(filtered_items)
                usernum = max(u, usernum)

        itemnum = Anum if domain == 'A' else Bnum
        self.user_train = user_train
        self.user_valid = user_valid
        self.usernum = usernum
        self.itemnum = itemnum

    def _train_valid_split(self, items):
        train_items = items[:-1]
        val_items = (items[:-1], items[-1])
        return train_items, val_items



class RecDataset(Dataset):
    def __init__(self, User, usernum, itemnum, item_set, aug_cands=None, maxlen=50):
        self.User = User
        self.usernum = usernum
        self.itemnum = itemnum
        self.item_set = item_set  # ✅ Pass item_set dict {asin: index}
        self.aug_cands = aug_cands
        self.maxlen = maxlen

    def __getitem__(self, index):
        user_id = list(self.User.keys())[index]
        seq_data = self.User[user_id]

        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)

        nxt = self.item_set[seq_data[-1]]  # ✅ Map ASIN to index
        idx = self.maxlen - 1
        ts = set([self.item_set[i] for i in seq_data])

        for i in reversed(seq_data[:-1]):
            seq[idx] = self.item_set[i]  # ✅ Map ASIN to index
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = self.item_set[i]
            idx -= 1
            if idx == -1:
                break

        return user_id, seq, pos, neg

    def __len__(self):
        return len(self.User)


    def __len__(self):
        return len(self.User)
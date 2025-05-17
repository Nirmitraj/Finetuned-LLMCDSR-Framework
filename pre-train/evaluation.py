from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import pickle
import os

class Evaluator(object):
    def __init__(self, dataset, itemnum, max_len, device, batch_size=128, dataset_name='pet-beauty', domain='B'):
        self.dataset = dataset
        self.itemnum = itemnum
        self.maxlen = max_len
        self.device = device
        self.batch_size = batch_size

        # ✅ Load item_set to map ASIN → index
        item_set_path = f'../data/{dataset_name}/item_set_{domain}.pkl'
        if not os.path.exists(item_set_path):
            raise FileNotFoundError(f"Cannot find item_set file at {item_set_path}")
        with open(item_set_path, 'rb') as f:
            self.asin2idx = pickle.load(f)

        self.all_neg_pool = np.zeros((len(self.dataset), self.itemnum - 1), dtype=np.int64)
        print("Generating negative pool...")

        row_idx = 0
        valid_rows = []
        for record in tqdm(self.dataset, total=len(self.dataset)):
            his_asin, target_asin = record

            # ✅ Map target ASIN → index
            target = self.asin2idx.get(target_asin, None)
            if target is None or target < 1 or target > self.itemnum:
                continue  # Skip invalid target

            item_idx = [target]
            pool = list(set(range(1, self.itemnum + 1)) - set(item_idx))

            if len(pool) != self.itemnum - 1:
                continue

            self.all_neg_pool[row_idx] = pool
            valid_rows.append((his_asin, target))
            row_idx += 1

        self.dataset = valid_rows
        self.all_neg_pool = self.all_neg_pool[:row_idx]

    def __call__(self, model):
        NDCG = defaultdict(float)
        HR = defaultdict(float)
        valid_user = 0.0
        cutoff_list = [10, 5, 3]

        all_his, all_cand = [], []

        neg_index = np.random.randint(0, self.all_neg_pool.shape[1], size=(self.all_neg_pool.shape[0], 999))
        all_neg = self.all_neg_pool[np.arange(self.all_neg_pool.shape[0])[:, None], neg_index]

        for user_id, (his_asins, target_idx) in tqdm(enumerate(self.dataset), disable=True):
            seq = np.zeros([self.maxlen], dtype=np.int32)
            idx = self.maxlen - 1

            # ✅ Map user's history ASINs to indices
            his_idxs = [self.asin2idx.get(asin, None) for asin in his_asins]
            his_idxs = [i for i in his_idxs if i is not None]

            if len(his_idxs) == 0:
                continue  # Skip if history mapping failed

            for i in reversed(his_idxs):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            item_idx = [target_idx]
            valid_user += 1

            neg_cands = all_neg[user_id].tolist()
            item_idx += neg_cands
            seq = seq.tolist()
            all_his.append(seq)
            all_cand.append(item_idx)

        for i in tqdm(range(0, len(all_his), self.batch_size), disable=True):
            seq_batch = all_his[i:i + self.batch_size]
            item_idx_batch = all_cand[i:i + self.batch_size]

            seq_tensor = torch.LongTensor(seq_batch).to(self.device)
            item_idx_tensor = torch.LongTensor(item_idx_batch).to(self.device)
            user_tensor = torch.arange(seq_tensor.shape[0]).to(self.device)

            predictions = -model.predict(user_tensor, seq_tensor, item_idx_tensor)

            rank_list = predictions.argsort(dim=-1).argsort(dim=-1)[:, 0].tolist()
            for rank in rank_list:
                for cutoff in cutoff_list:
                    if rank < cutoff:
                        NDCG[cutoff] += 1 / np.log2(rank + 2)
                        HR[cutoff] += 1

        for cutoff in cutoff_list:
            NDCG[cutoff] /= valid_user
            HR[cutoff] /= valid_user

        return NDCG, HR

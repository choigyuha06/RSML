import torch
import random
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

def set_seed(seed):
    """
    재현성을 위해 random, numpy, torch의 시드를 고정하는 함수
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    # CUDNN을 사용한 가속을 포기하고 재현성을 확보
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocessing(data_path):
    files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    user_map, item_map = {}, {}
    current_user_id, current_item_id = 0, 0
    interactions = []
    for file_path in files:
        with open(file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
            item_id = None
            for line in tqdm(lines, desc=f"Processing {os.path.basename(file_path)}", total=len(lines)):
                if line.strip().endswith(':'):
                    movie_id_raw = int(line.split(':')[0])
                    if movie_id_raw not in item_map:
                        item_map[movie_id_raw] = current_item_id
                        current_item_id += 1
                    item_id = item_map[movie_id_raw]
                else:
                    user_id_raw, _, _ = line.partition(',')
                    user_id_raw = int(user_id_raw)
                    if user_id_raw not in user_map:
                        user_map[user_id_raw] = current_user_id
                        current_user_id += 1
                    user_id = user_map[user_id_raw]
                    interactions.append((user_id, item_id))
    print(f"전처리 완료. 총 사용자 수: {len(user_map)}, 총 아이템 수: {len(item_map)}, 총 상호작용 수: {len(interactions)}")
    user_interactions = defaultdict(set)
    for u, i in interactions:
        user_interactions[u].add(i)
    return interactions, user_map, item_map, user_interactions


def negative_sampling(user, item_list: set, user_item_interactions, n_samples=1):
    pos_list = user_item_interactions[user]

    neg_list = list(item_list - pos_list)

    return random.sample(neg_list, n_samples)


def constrict_triplet(interactions, user_item_interactions, n_samples):
    item_list = set().union(*user_item_interactions.values())
    user_negative_samples = set()

    for u in tqdm(user_item_interactions.keys(), desc="constructing negative samples"):
        user_negative_samples[u] = negative_sampling(u, item_list, user_item_interactions, n_samples)

    triplets = []
    for u, v in tqdm(interactions, desc="Constructing triplets"):
        for neg_v in user_negative_samples[u]:
            triplets.append([u, v, neg_v])

    return triplets

def collate_fn_eval(batch):
    users = torch.stack([item[0] for item in batch])
    true_items = [item[1] for item in batch]
    padded_true_items = pad_sequence(true_items, batch_first=True, padding_value=-1)
    return users, padded_true_items
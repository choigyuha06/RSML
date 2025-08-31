import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import negative_sampling


class NetflixTripletDataset2(Dataset):
    def __init__(self, interactions, num_items, user_interactions):
        self.triplets = []
        item_list = set().union(*user_interactions.values())
        user_negative_samples = {}

        for u in tqdm(user_interactions.keys(), desc="constructing negative samples"):
            user_negative_samples[u] = negative_sampling(u, item_list, user_interactions, n_samples=1)

        for user_id, pos_item_id in tqdm(interactions, desc="construting triplets"):
            for neg_item_id in user_negative_samples[user_id]:
                self.triplets.append([user_id, pos_item_id, neg_item_id])


    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


class EvalDataset(Dataset):
    def __init__(self, user_item_map):
        self.users = sorted(list(user_item_map.keys()))
        self.user_item_map = user_item_map
    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        user = self.users[idx]
        return torch.tensor(user), torch.tensor(list(self.user_item_map[user]))
#3-Step

import pickle
import numpy as np
import argparse
from tqdm import tqdm
from os.path import join as path_join
from sentence_transformers import SentenceTransformer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', required=True)  # e.g., /data/music-videogame/
    parser.add_argument('--domain', default='A', type=str)  # A or B
    parser.add_argument('--exp_name', default='default', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--model_path', required=True, type=str)  # jina embedding model
    parser.add_argument('--save_path', required=True, type=str)  # where to save the .npy
    return parser.parse_args()


def get_data(task_path, domain):
    data_dict = {}
    with open(path_join(task_path, f'text_{domain}.pkl'), 'rb') as f:
        text_descriptions = pickle.load(f)
    with open(path_join(task_path, f'item_set_{domain}.pkl'), 'rb') as f:
        id_index_map = pickle.load(f)
    for item_id, index in id_index_map.items():
        text = text_descriptions[item_id]
        data_dict[index] = text
    data = [data_dict[i] for i in range(len(data_dict))]  # sorted by index
    return data


def prepare_model(model_path, device):
    model = SentenceTransformer(model_path, device=device)
    return model


def runner(data, model, batch_size):
    all_embeddings = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embeddings)
    return np.concatenate(all_embeddings, axis=0)


def save_mat(mat, save_path, domain):
    np.save(path_join(save_path, f'{domain}_item_jina.npy'), mat)


def main(args):
    data = get_data(args.task_path, args.domain)
    model = prepare_model(args.model_path, args.device)
    mat = runner(data, model, args.batch_size)
    save_mat(mat, args.save_path, args.domain)


if __name__ == "__main__":
    args = get_args()
    main(args)

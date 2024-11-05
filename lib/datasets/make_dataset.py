import ast
import os
from time import time

import torch
import torch.utils.data
from openai import AzureOpenAI
from torch.utils.data.dataloader import default_collate

from lib.utils import logger
from lib.utils.registry import Registry

DATASET = Registry('dataset')
from .humanise.humanise_motion import HumaniseMotion

client = AzureOpenAI(
            azure_deployment="openai-base-demo-4o",
            api_version='2024-04-01-preview',
            api_key="35850b4269124a18ae47bde5f0a9c926",
            azure_endpoint="https://openai-base-demo.openai.azure.com",
        )

def make_dataset(cfg, split='train'):
    tic = time()
    dat_cfg = cfg.get(f"{split}_dat")
    logger.info(f"Making {split} dataset: {dat_cfg.name}")

    # load real dataset
    dataset = DATASET.get(dat_cfg.name)(cfg.dat_cfg, dat_cfg.split)
    limit_size = dat_cfg.limit_size
    if limit_size > 0 and len(dataset) > limit_size:
        logger.warning(f"Working on subset of size {limit_size}")
        dataset = torch.utils.data.Subset(dataset, list(range(limit_size)))
        

    logger.debug(f"Time for making dataset: {time() - tic:.2f}s")
    return dataset


def classify_action(utterance):
    prompt = f"Classify the action as one of the following: sit, walk, lie, standup. Utterance: '{utterance}'"
    # Use OpenAI's model to determine the action based on the utterance
    response = client.chat.completions.create(
        model="text-davinci-003",
        messages=[{
            'role': 'system',
            'content': prompt
        }],
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0
    )
    action = response.choices[0].message.content
    return action if action in ['sit', 'walk', 'lie', 'standup'] else "unknown"


def fetch_vocalized_text():
    response_txt_filepath = '/content/drive/MyDrive/Research_v2/out/locate/gt_chatgpt_paper/response/0.txt'
    if os.path.exists(response_txt_filepath):
        with open(response_txt_filepath, 'r') as f:
            lines = f.readlines()
        
        # Search for the line containing the dictionary-like structure
        for line in lines:
            if 'input_text' in line:
                # Attempt to parse the line as a dictionary
                try:
                    data_dict = ast.literal_eval(line.strip())
                    # Extract 'input_text' if it exists in the dictionary
                    input_text_value = data_dict.get('input_text', 'input_text not found')
                    print(f"input_text value is: {input_text_value}")
                    return input_text_value
                except (ValueError, SyntaxError) as e:
                    print(f"Failed to parse line as dictionary: {e}")
                    return None
        
        print("input_text not found in file.")
        return None
    else:
        print('File does not exist')
        return None


def collate_fn_wrapper(batch):
    text = fetch_vocalized_text()
    keys_to_collate_as_list = ['meta']
    list_in_batch = {}
    
    for k in keys_to_collate_as_list:
        if k in batch[0]:
            list_in_batch[k] = [data[k] for data in batch]
            for meta in list_in_batch[k]:
                meta['utterance'] = text
                meta['action'] = classify_action(text)
    
    # Use default collate for the rest of the batch
    batch = default_collate(batch)
    batch.update({k: v for k, v in list_in_batch.items()})

    return batch


def make_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        return torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
    else:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    return sampler


def make_data_loader(cfg, split='train'):
    dataset = make_dataset(cfg, split)
    logger.info(f"Final {split} dataset size: {len(dataset)}")

    datloader_cfg = cfg.get(split)
    batch_size = datloader_cfg.batch_size
    num_workers = datloader_cfg.num_workers

    sampler = make_data_sampler(dataset, datloader_cfg.shuffle, cfg.distributed)

    # assume 1*node with N*Gpus: evenly adjust batchsize and num_workers
    if cfg.distributed:
        assert batch_size % int(os.environ['WORLD_SIZE']) == 0
        batch_size = batch_size // int(os.environ['WORLD_SIZE'])
        num_workers = num_workers // int(os.environ['WORLD_SIZE'])

    collate_fn = collate_fn_wrapper
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers= (split == 'train' and num_workers > 0),
        collate_fn=collate_fn,
    )

    return dataloader
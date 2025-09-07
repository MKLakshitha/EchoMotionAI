import ast
import os
from time import time
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.utils.data
from openai import AzureOpenAI
from torch.utils.data.dataloader import default_collate

from lib.utils import logger
from lib.utils.registry import Registry

DATASET = Registry('dataset')
from .humanise.humanise_motion import HumaniseMotion
def load_config():
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'configurations.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()


client = AzureOpenAI(
            azure_deployment=config['azure_openai']['deployment'],
            api_version=config['azure_openai']['api_version'],
            api_key=config['azure_openai']['api_key'],
            azure_endpoint=config['azure_openai']['endpoint']
        )


def make_dataset(cfg, split='train'):
    tic = time()


    dat_cfg = cfg.get(f"{split}_dat")
    logger.info(f"Making {split} dataset: {dat_cfg.name}")
    print(f" scene id is {dat_cfg.scene_id}")
    # load real dataset
    dataset = DATASET.get(dat_cfg.name)(cfg.dat_cfg, dat_cfg.split)
    print(f"Dataset {dat_cfg.name} loaded with {len(dataset)} samples and dataset is {dataset}")
    limit_size = dat_cfg.limit_size
    if limit_size > 0 and len(dataset) > limit_size:
        logger.warning(f"Working on subset of size {limit_size}")
        dataset = torch.utils.data.Subset(dataset, list(range(limit_size)))
        

    logger.debug(f"Time for making dataset: {time() - tic:.2f}s")
    return dataset


def classify_action(utterance):
    prompt = f"""Classify the action as one of the following: sit, walk, lie, standup. Utterance: '{utterance}'
    if similiar words like 'go' should be classified as 'walk' , 'stand' should be classified as 'standup', 'sleep' should be classified as 'lie', 'sit down' should be classified as 'sit'.
    Only Respond with the action name without any additional text.
    """
    # Use OpenAI's model to determine the action based on the utterance
    response = client.chat.completions.create(
        model="",
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
    print(f"Action classified as: {action}")
    return action if action in ['sit', 'walk', 'lie', 'standup'] else "unknown"


def fetch_vocalized_text(cfg,response_folder):
    response_txt_filepath = f'{response_folder}/{cfg.data_id}.txt'
    
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
                    print(f"\ninput_text value is: {input_text_value}")
                    return input_text_value
                except (ValueError, SyntaxError) as e:
                    print(f"Failed to parse line as dictionary: {e}")
                    return None
        
        print("input_text not found in file.")
        return None
    else:
        print('File does not exist')
        return None


def collate_fn_wrapper(cfg,batch,response_folder):
    text = fetch_vocalized_text(cfg,response_folder)
    keys_to_collate_as_list = ['meta']
    list_in_batch = {}
    
    for k in keys_to_collate_as_list:
        if k in batch[0]:
            list_in_batch[k] = [data[k] for data in batch]
            for meta in list_in_batch[k]:
                meta['utterance'] = text
                meta['action'] = classify_action(text)
    
    
    
    def safe_to_tensor(value):
        try:
            if isinstance(value, np.ndarray):
                # Log array info for debugging
                logger.debug(f"Converting numpy array with dtype {value.dtype} and shape {value.shape}")
                if np.issubdtype(value.dtype, np.floating):
                    return torch.from_numpy(value.astype(np.float32))
                elif np.issubdtype(value.dtype, np.integer):
                    return torch.from_numpy(value.astype(np.int64))
                else:
                    return torch.from_numpy(value.astype(np.float32))
            elif isinstance(value, (int, float)):
                return torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to tensors if they contain numbers
                if all(isinstance(x, (int, float)) for x in value):
                    return torch.tensor(value, dtype=torch.float32)
            return value
        except Exception as e:
            logger.error(f"Error converting value {type(value)}: {e}")
            return value

    # Custom collation for numpy arrays
    processed_batch = []
    for item_idx, item in enumerate(batch):
        processed_item = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                processed_item[key] = value
            elif isinstance(value, dict):
                processed_item[key] = value
            else:
                # Log problematic values for debugging
                logger.debug(f"Processing item {item_idx}, key {key}, type {type(value)}")
                processed_item[key] = safe_to_tensor(value)
        processed_batch.append(processed_item)
    
    # Custom collate function to handle tensors of different types
    def custom_collate(batch):
        if len(batch) == 0:
            return batch
        
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            try:
                return torch.stack(batch, 0)
            except:
                return batch
        elif isinstance(elem, (int, float)):
            return torch.tensor(batch)
        elif isinstance(elem, dict):
            return {key: custom_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, (tuple, list)) and len(elem) > 0:
            try:
                return type(elem)(custom_collate(samples) for samples in zip(*batch))
            except:
                return batch
        else:
            return batch

    # Use custom collate for the processed batch
    try:
        batch = custom_collate(processed_batch)
        batch.update({k: v for k, v in list_in_batch.items()})
        return batch
    except Exception as e:
        logger.error(f"Error in final collation: {e}")
        # Fall back to default collate if custom collate fails
        try:
            batch = default_collate(processed_batch)
            batch.update({k: v for k, v in list_in_batch.items()})
            return batch
        except Exception as e2:
            logger.error(f"Both custom and default collate failed: {e2}")
            raise


def make_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        return torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
    else:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    return sampler


def make_data_loader(cfg, response_folder,split='train'):
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
    
    collate_fn = lambda batch: collate_fn_wrapper(cfg,batch, response_folder)

    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers= (split == 'train' and num_workers > 0),
        collate_fn=collate_fn,
    )

    return dataloader
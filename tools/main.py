import argparse
import os
import pickle

from tqdm import tqdm

from lib.config import make_cfg
from lib.datasets.make_dataset import make_dataset,make_data_loader
from lib.localization.eval_localization import evaluate_pd_bbox_by_span
from lib.localization.scene_graph import SceneGraph
from lib.utils.comm import seed_everything
from lib.networks.make_network import make_network
from lib.utils.net_utils import load_network, to_cuda
from pathlib import Path
from lib.utils import offscreen_flag

from lib.wrapper.two_stage import save_sample
from lib.wrapper.wrapper import TwoStageWrapper
import torch



def locate_target(cfg):
    scene_graph = SceneGraph(cfg)
    for iter, data in enumerate(tqdm(data_loader)):
        response_file = f'{response_folder}/{iter}.txt'
        object_points_file = f'{object_points_folder}/{iter}.pkl'
        
        if os.path.exists(response_file): continue

        data['pos'] = data[f'xyz_hm']

        pred_center, pred_points, response_objects, response_relations,text = \
            scene_graph.inference(data)
        
        tgt_center = data['obj_center_hm'].numpy()
        tgt_points = data['xyz_hm'][data['obj_mask']].numpy()
        
        with open(response_file, 'w') as f:
            f.write(f'tgt_center: {tgt_center}\n')
            f.write(f'pred_center: {pred_center}\n')
            f.write(f'{response_objects}\n')
            f.write(f'{response_relations}\n')
            f.write(f'text: {text}\n')
        
        with open(object_points_file, 'wb') as f:
            pickle.dump({
                'tgt_center': tgt_center,
                'tgt_points': tgt_points,
                'pred_center': pred_center,
                'pred_points': pred_points,
                'text': text
            }, f)


def locate_eval(cfg):
    center_loss = []
    all_count, all_found = 0, 0

    for iter, data in enumerate(tqdm(data_loader)):
        object_points_file = f'{object_points_folder}/{iter}.pkl'
        if os.path.exists(object_points_file):
            with open(object_points_file, 'rb') as f:
                pdata = pickle.load(f)
        
            gt_points = pdata['tgt_points']
            pd_points = pdata['pred_points']
            center_loss.append(((pdata['pred_center'] - pdata['tgt_center']) ** 2).sum() ** 0.5)
            
            found = evaluate_pd_bbox_by_span(pd_points, gt_points, cfg.mode)
            all_count += 1
            all_found += found
    
    print('all: ', all_count)
    print('acc: ', all_found / all_count)
    print('center dist.: ', sum(center_loss) / all_count)


def process_pred_object(batch):
    coord = cfg.coord
    obj_mask_list, obj_center_hm_list, t_to_scannet_list, xyz_oc_list  = [], [], [], []
    for b in range(len(batch['meta'])):
        data_id = batch['meta'][b]['unique_idx']
        with open(f"{cfg.pred_center_root}/{cfg.data_id}.pkl", 'rb') as f:
            od = pickle.load(f)
        
        scene_points = batch['xyz_hm'][b]
        pred_points = od['pred_points'] # bbx points
        bbx_min = pred_points.min(0)
        bbx_max = pred_points.max(0)
        obj_mask = (scene_points[:, 0] > bbx_min[0]) & (scene_points[:, 0] < bbx_max[0]) & (scene_points[:, 1] > bbx_min[1]) & (scene_points[:, 1] < bbx_max[1]) & (scene_points[:, 2] > bbx_min[2]) & (scene_points[:, 2] < bbx_max[2])
        obj_center_hm = torch.mean(scene_points[obj_mask], dim=0)
        
        if coord == 'oc':
            t_hm2oc = - obj_center_hm
            t_to_scannet = - batch['t_sn2hm'][b] - t_hm2oc
            xyz_oc = batch['xyz_hm'][b] + t_hm2oc
            t_to_scannet_list.append(t_to_scannet)
            xyz_oc_list.append(xyz_oc)
        
        obj_mask_list.append(obj_mask)
        obj_center_hm_list.append(obj_center_hm)
    
    batch['gt'] = {
        'obj_mask': batch['obj_mask'].clone(),
        'obj_center_hm': batch['obj_center_hm'].clone(),
    }
    batch['obj_mask'] = torch.stack(obj_mask_list, dim=0)
    batch['obj_center_hm'] = torch.stack(obj_center_hm_list, dim=0)

    if coord == 'oc':
        batch['gt'].update({
            't_to_scannet': batch['t_to_scannet'].clone(),
            'xyz_oc': batch['xyz_oc'].clone(),
        })
        batch['t_to_scannet'] = torch.stack(t_to_scannet_list, dim=0)
        batch['xyz_oc'] = torch.stack(xyz_oc_list, dim=0)


def run_network(network):
    if cfg.save:
        save_dict = []
        output_folder = Path(cfg.record_dir) / f'{cfg.task}_{cfg.save_type}_{cfg.action}'
        print('making outputfolder')
        os.makedirs(output_folder, exist_ok=True)
    
    for iter, batch in enumerate(tqdm(data_loader)):
        # setup data
        batch = to_cuda(batch)
        # center
        if cfg.use_pred_center:
            process_pred_object(batch)
        
        # run methods
        with torch.no_grad():
            network(batch, inference=True, compute_supervision=False, compute_loss=False)
        
        # save
        if cfg.save:
            save_dict.extend(save_sample(cfg, batch))
    
    if cfg.save:
        output_file = output_folder / f'sample.pkl'
        print(f'saving to {output_file}')
        with open(output_file, 'wb') as fp:
            pickle.dump(save_dict, fp)



def initialize_cfg(args):
    """Initializes and returns the configuration based on parsed arguments."""
    cfg = make_cfg(args)
    return cfg





def generate_results(cfg):
    def load_one_network(args):
        cfg = make_cfg(args)
        cfg.resume_model_dir = ''
        network = make_network(cfg)
        load_network(network, resume=True, cfg=cfg, epoch=-1)
        return network, cfg

    # load traj net
    traj_args = parser.parse_args(['-c', cfg.two_stage_cfg.traj, 'net_cfg.k_sample', str(cfg.two_stage_cfg.traj_k)])
    traj_network, traj_cfg = load_one_network(traj_args)
    # load motion net
    motion_args = parser.parse_args(['-c', cfg.two_stage_cfg.motion, 'net_cfg.k_sample', str(cfg.two_stage_cfg.motion_k)])
    motion_network, motion_cfg = load_one_network(motion_args)
    # whole network
    network = TwoStageWrapper(traj_network, motion_network, cfg, traj_cfg, motion_cfg).eval().cuda()
    run_network(network)

if __name__ == '__main__':
    # Seed function
    seed_everything()

    # First Argument Parsing and Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", "-c",default='configs/locate/locate_chatgpt.yaml')
    parser.add_argument("--is_test", action="store_true", default=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = initialize_cfg(args)
    data_loader = make_dataset(cfg, split='test')

    out_folder = f'{cfg.record_dir}/{cfg.mode}_{cfg.method}_{cfg.prompt_type}'
    object_points_folder = f'{out_folder}/object_points'
    os.makedirs(object_points_folder, exist_ok=True)
    response_folder = f'{out_folder}/response'
    os.makedirs(response_folder, exist_ok=True)
    # Execution
    locate_target(cfg)
    locate_eval(cfg)


    del cfg
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", "-c", default='configs/test/generate.yaml')
    parser.add_argument("--is_test", action="store_true", default=False)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    arg = parser.parse_args()

    cfg = initialize_cfg(arg)
    print(arg)
    # Data setup with new configuration
    split = 'test'
    cfg.get(split).shuffle = False
    cfg.get(split).batch_size = 1
    data_loader = make_data_loader(cfg, response_folder,split=split)
    
    generate_results(cfg)





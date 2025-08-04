import argparse
import os
import pickle

from tqdm import tqdm
import numpy as np
from lib.config import make_cfg
from lib.datasets.make_dataset import make_dataset
from lib.localization.eval_localization import evaluate_pd_bbox_by_span
from lib.localization.scene_graph import SceneGraph
from lib.utils.comm import seed_everything


# Global scene_graph to maintain session persistence
persistent_scene_graph = None

def main(cfg):
    global persistent_scene_graph
    
    # Create scene_graph only once to maintain session persistence
    if persistent_scene_graph is None:
        persistent_scene_graph = SceneGraph(cfg)
        
    for iter, data in enumerate(tqdm(data_loader)):
        response_file = f'{response_folder}/{iter}.txt'
        object_points_file = f'{object_points_folder}/{iter}.pkl'
        
        if os.path.exists(response_file): continue

        data['pos'] = data[f'xyz_hm']

        # Use our persistent scene_graph with accumulated context
        pred_center, pred_points, response_objects, response_relations, text = \
            persistent_scene_graph.inference(data)
        
        tgt_center = data['obj_center_hm'].numpy()
        tgt_points = data['xyz_hm'][data['obj_mask']].numpy()
        
        with open(response_file, 'w') as f:
            f.write(f'tgt_center: {tgt_center.tolist()}\n')
            f.write(f'pred_center: {pred_center.tolist() if hasattr(pred_center, "tolist") else pred_center}\n')
            f.write(f'{response_objects}\n')
            f.write(f'{response_relations}\n')
            f.write(f'text: {text}\n')
        
        with open(object_points_file, 'wb') as f:
            pickle.dump({
                'tgt_center': tgt_center.tolist(),
                'tgt_points': tgt_points.tolist(),
                'pred_center': pred_center.tolist() if hasattr(pred_center, 'tolist') else pred_center,
                'pred_points': pred_points.tolist() if hasattr(pred_points, 'tolist') else pred_points,
                'text': text
            }, f)



def eval(cfg):
    center_loss = []
    all_count, all_found = 0, 0

    for iter, data in enumerate(tqdm(data_loader)):
        object_points_file = f'{object_points_folder}/{iter}.pkl'
        if os.path.exists(object_points_file):
            with open(object_points_file, 'rb') as f:
                pdata = pickle.load(f)
        
            # Convert lists back to numpy arrays
            gt_points = np.array(pdata['tgt_points'])
            pd_points = np.array(pdata['pred_points'])
            pred_center = np.array(pdata['pred_center'])
            tgt_center = np.array(pdata['tgt_center'])
            
            # Calculate center loss using numpy arrays
            center_loss.append(((pred_center - tgt_center) ** 2).sum() ** 0.5)
            
            found = evaluate_pd_bbox_by_span(pd_points, gt_points, cfg.mode)
            all_count += 1
            all_found += found
    
    print('all: ', all_count)
    print('acc: ', all_found / all_count)
    print('center dist.: ', sum(center_loss) / all_count)


if __name__ == '__main__':
    seed_everything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", "-c", type=str, required=True)
    parser.add_argument("--is_test", action="store_true", default=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = make_cfg(args)

    data_loader = make_dataset(cfg, split='test')
    
    # log folders
    out_folder = f'{cfg.record_dir}/{cfg.mode}_{cfg.method}_{cfg.prompt_type}'
    object_points_folder = f'{out_folder}/object_points'
    os.makedirs(object_points_folder, exist_ok=True)
    response_folder = f'{out_folder}/response'
    os.makedirs(response_folder, exist_ok=True)

    main(cfg)
    eval(cfg)
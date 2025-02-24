import json
from pathlib import Path
import yaml

import azure.cognitiveservices.speech as speechsdk
import numpy as np
from shapely.geometry import MultiPoint
from sklearn.neighbors import NearestNeighbors

from lib.localization.chatgpt_talker import ChatGPTTalker


def load_config():
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'configurations.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def transcribe_audio_to_text():
    config = load_config()
    # Set up the Azure Speech configuration
    speech_config = speechsdk.SpeechConfig(
        subscription=config['azure_speech']['subscription_id'],
        region=config['azure_speech']['region']
    )
    speech_config.speech_recognition_language = config['azure_speech']['language']

    # Get the audio file path and wait for it
    audio_file = Path("audio_input/input.wav")
    import time
    max_wait_time = 600  # Maximum wait time in seconds (10 minutes)
    start_time = time.time()
    min_file_size = 44  # Minimum WAV file header size in bytes
    
    print("Waiting for audio file to be available...")
    while True:
        if time.time() - start_time > max_wait_time:
            print(f"Timeout: No valid audio file found at {audio_file.absolute()} after {max_wait_time} seconds")
            return "No audio file found. Using default text: where is the chair"
            
        if not audio_file.exists():
            time.sleep(1)
            continue
            
        # Check if file size is growing
        current_size = audio_file.stat().st_size
        if current_size <= min_file_size:
            print("Waiting for audio file to be written...")
            time.sleep(1)
            continue
            
        # Wait a bit more to ensure file is completely written
        time.sleep(0.5)
        new_size = audio_file.stat().st_size
        if new_size == current_size and new_size > min_file_size:
            print(f"Audio file found and ready at {audio_file.absolute()} (size: {new_size} bytes)")
            break
        
        time.sleep(0.5)

    # Configure to recognize speech from an audio file
    audio_config = speechsdk.AudioConfig(filename=str(audio_file))
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Processing audio file...")
    
    # Use a done event to signal when recognition is complete
    done = False
    final_transcription = ""

    def handle_result(evt):
        nonlocal final_transcription
        if evt.result.text:
            final_transcription += evt.result.text + " "

    def stop_cb(evt):
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the speech recognizer
    recognizer.recognized.connect(handle_result)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    recognizer.start_continuous_recognition()
    while not done:
        pass
    recognizer.stop_continuous_recognition()

    if not final_transcription:
        return "No speech detected. Using default text: where is the chair"

    return final_transcription.strip()


class SceneGraph():
    def __init__(self, cfg) -> None:
        self.mode = cfg.mode
        self.talker = ChatGPTTalker(
            prompt_type=cfg.prompt_type)
       
        self.id2obj = {}
        if self.mode == 'gt':
            self.scannet_root = Path(cfg.dat_cfg.scannet_root) / 'scans'
        else:
            raise NotImplementedError
       
        # constants
        self.min_above_below_distance = 0.06
        self.min_to_be_above_below_area_ratio = 0.2
        self.occ_thresh = 0.5
        self.min_forbidden_occ_ratio = 0.1
        self.intersect_ratio_thresh = 0.1
        pass
 
    def inference(self, batch):
        scene_id = batch['meta']['scene_id']
        text = transcribe_audio_to_text()
        print(f'Text is {text}')
        if self.mode == 'gt':
            self.load_gt_scene(batch)
       
        target_object, anchor_objects, response_objects = \
            self.talker.ask_objects(text, self.id2obj[scene_id])
       
        if len(anchor_objects) == 0:
            pred_center, pred_points = self.get_obj_center(self.id2obj[scene_id], target_object)
            return pred_center, pred_points, response_objects, None, text
        else:
            relations = self.scenegraph_relationship(scene_id, target_object, anchor_objects)
            target_name, response_relations = \
                self.talker.ask_relations(text, relations, self.id2obj[scene_id], target_object, anchor_objects)
            pred_center, pred_points = self.get_obj_center(self.id2obj[scene_id], target_name)
            return pred_center, pred_points, response_objects, response_relations,text
   
 
    def get_obj_center(self, label_objects, target_name):
        for label, objects in label_objects.items():
            if target_name == label:
                return objects[0]['center'], objects[0]['verts']
            for obj in objects:
                if target_name == obj['name']:
                    return obj['center'], obj['verts']
   
   
    def load_gt_scene(self, batch):
        scene_id = batch['meta']['scene_id']
        if scene_id in self.id2obj:
            return
        all_objects = {}
        scannet_scan_root = self.scannet_root / scene_id
        print(f"Located Inference load gt scene scannet root is :{scannet_scan_root}")
        # should occur an error need to change name of the below file..
        # aggr_file= list(scannet_scan_root.glob("*_vh_clean.aggregation.json"))[0]
        aggr_file= list(scannet_scan_root.glob("*aggregation.json"))[0]
        with open(aggr_file, 'r') as f:
            aggr_data = json.load(f)
        for i, seg in enumerate(aggr_data['segGroups']):
            if seg["label"] in ['wall', 'floor', 'ceiling']:
                continue
            obj_name = seg["label"]
            obj_verts = batch['pos'][batch['obj_label'] == seg['objectId']]
            obj_dict = {
                'id': seg['objectId'],
                'name': f"{obj_name} {seg['objectId']}",
                'verts': obj_verts.numpy(),
                'center': obj_verts.mean(0).numpy(),
                'bbx_max': obj_verts.max(0)[0].numpy(),
                'bbx_min': obj_verts.min(0)[0].numpy(),
            }
            if obj_name in all_objects:
                all_objects[obj_name].append(obj_dict)
            else:
                all_objects[obj_name] = [obj_dict]
        self.id2obj[scene_id] = all_objects
   
   
    def scenegraph_relationship(self, scene_id, target_object, anchor_objects):
        # build relationship
        relations = {}
        targets = self.id2obj[scene_id][target_object]
        anchors = [self.id2obj[scene_id][obj][0] for obj in anchor_objects]
        if len(anchors) == 1:
            relations.update(self.horizontal_relationship(targets, anchors[0]))
            relations.update(self.vertical_relationship(targets, anchors[0]))
        else:
            NUM = len(anchors)
            for j in range(NUM):
                for k in range(j+1, NUM):
                    relations.update(self.between_relationship(targets, anchors[j], anchors[k]))
            for anchor in anchors:
                relations.update(self.horizontal_relationship(targets, anchor))
                relations.update(self.vertical_relationship(targets, anchor))
        return relations
 
    def horizontal_relationship(self, targets, anchor):
        dist2anchor = []
        for target in targets:
            dist2anchor.append(dist_between_points(target['verts'][:, :2], anchor['verts'][:, :2]))
        min_idx = np.argmin(np.array(dist2anchor))
        near_target = targets[min_idx]["name"]
        max_idx = np.argmax(np.array(dist2anchor))
        far_target = targets[max_idx]["name"]
        relations = {
            near_target: f'{near_target} is near to {anchor["name"]}',
            far_target: f'{far_target} is far from {anchor["name"]}'
        }
        return relations
       
    def vertical_relationship(self, targets, anchor):
        relations = {}
        for target in targets:
            if iou_2d(target, anchor) < 0.001:  # No intersection at all (not in the vicinty of each other)
                continue
 
            target_bottom_anchor_top_dist = target['verts'].min(0)[2] - anchor['verts'].max(0)[2]
 
            target_above_anchor = target_bottom_anchor_top_dist > self.min_above_below_distance
            target_below_anchor = -target_bottom_anchor_top_dist > self.min_above_below_distance
 
            if target_above_anchor:
                relations.update({
                    target['name']: f'{target["name"]} is above {anchor["name"]}'
                })
            elif target_below_anchor:
                relations.update({
                    target['name']: f'{target["name"]} is below {anchor["name"]}'
                })
        return relations
   
    def between_relationship(self, targets, anc_a, anc_b):
        relations = {}
        for target in targets:
            anchor_a_points = tuple(map(tuple, anc_a["verts"][:, :2]))  # x, y coordinates
            anchor_b_points = tuple(map(tuple, anc_b["verts"][:, :2]))
            target_points = tuple(map(tuple, target["verts"][:, :2]))
 
            if is_between(
                anc_a_points=anchor_a_points,
                anc_b_points=anchor_b_points,
                target_points=target_points,
                occ_thresh=self.occ_thresh,
                intersect_ratio_thresh=self.intersect_ratio_thresh):
                relations.update({
                    target["name"]: f'{target["name"]} is between {anc_a["name"]} and {anc_b["name"]}',
                })
        return relations
   
   
def dist_between_points(points1, points2):
    nn = NearestNeighbors(n_neighbors=1).fit(points1)
    dists, _ = nn.kneighbors(points2)
    return np.min(dists)
 
def iou_2d(a, b):
    box_a = [a['bbx_min'][0], a['bbx_min'][1], a['bbx_max'][0], a['bbx_max'][1]]
    box_b = [b['bbx_min'][0], b['bbx_min'][1], b['bbx_max'][0], b['bbx_max'][1]]
 
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
 
    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA) * max(0, yB - yA)
 
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
 
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou
 
def is_between(
    anc_a_points: tuple,
    anc_b_points: tuple,
    target_points: tuple,
    occ_thresh: float,
    intersect_ratio_thresh: float):
    """
    Check whether a target object lies in the convex hull of the two anchors.
    @param anc_a_points: The vertices of the first anchor's 2d top face.
    @param anc_b_points: The vertices of the second anchor's 2d top face.
    @param target_points: The vertices of the target's 2d top face.
    @param occ_thresh: By considering the target intersection ratio with the convexhull of the two anchor,
    which is calculated by dividing the target intersection area to the target's area, if the ratio is
    bigger than the occ_thresh, then we consider this target is between the two anchors.
    @param min_forbidden_occ_ratio: used to create a range of intersection area ratios wherever any target
    object occupies the convexhull with a ratio within this range, we consider this case is ambiguous and we
    ignore generating between references with such combination of possible targets and those two anchors
    @param target_anchor_intersect_ratio_thresh: The max allowed target-to-anchor intersection ratio, if the target
    is intersecting with any of the anchors with a ratio above this thresh, we should ignore generating between
    references for such combinations
 
    @return: (bool) --> (target_lies_in_convex_hull_statisfying_constraints)
    """
    # Get the convex hull of all points of the two anchors
    convex_hull = MultiPoint(anc_a_points + anc_b_points).convex_hull
 
    # Get anchor a, b polygons
    polygon_a = MultiPoint(anc_a_points).convex_hull
    polygon_b = MultiPoint(anc_b_points).convex_hull
    polygon_t = MultiPoint(target_points).convex_hull
 
    # Candidate should fall completely/with a certain ratio in the convex_hull polygon
    occ_ratio = convex_hull.intersection(polygon_t).area / polygon_t.area
    if occ_ratio < occ_thresh:  # The object is not in the convex-hull enough to be considered between
        return False
 
    # Candidate target should never be intersecting any of the anchors
    if polygon_t.intersection(polygon_a).area / polygon_t.area > intersect_ratio_thresh or \
       polygon_t.intersection(polygon_b).area / polygon_t.area > intersect_ratio_thresh: return False
 
    return True

import json
from pathlib import Path
import yaml
import logging
import datetime
import pymongo
import numpy as np
from pymongo import MongoClient

import azure.cognitiveservices.speech as speechsdk
import numpy as np
from shapely.geometry import MultiPoint
from sklearn.neighbors import NearestNeighbors

from lib.localization.chatgpt_talker import ChatGPTTalker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # Static class variable to maintain session ID across instances
    _persistent_session_id = None
    
    def __init__(self, cfg) -> None:
        self.mode = cfg.mode
        self.talker = ChatGPTTalker(
            prompt_type=cfg.prompt_type)
       
        self.id2obj = {}
        if self.mode == 'gt':
            self.scannet_root = Path(cfg.dat_cfg.scannet_root) / 'scans'
        else:
            raise NotImplementedError
            
        # Register exit handler to properly close sessions
        import atexit
        atexit.register(self.close_session)
       
        # constants
        self.min_above_below_distance = 0.06
        self.min_to_be_above_below_area_ratio = 0.2
        self.occ_thresh = 0.5
        self.min_forbidden_occ_ratio = 0.1
        self.intersect_ratio_thresh = 0.1
        
        # Initialize MongoDB connection
        config = load_config()
        try:
            self.mongo_client = MongoClient(config['mongodb']['connection_string'])
            self.db = self.mongo_client[config['mongodb']['database_name']]
            self.conversations_collection = self.db[config['mongodb']['collections']['conversations']]
            
            # Use existing session ID if available, otherwise create a new one
            if SceneGraph._persistent_session_id is None:
                # Generate a unique session ID based on timestamp
                SceneGraph._persistent_session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                logger.info(f"Created new persistent session ID: {SceneGraph._persistent_session_id}")
            else:
                logger.info(f"Reusing existing session ID: {SceneGraph._persistent_session_id}")
                
            # Set instance session ID from class variable
            self.session_id = SceneGraph._persistent_session_id
            
            # Clear any hanging sessions that might not have been properly closed
            self._cleanup_old_sessions()
            
            # Store session start event
            self.db[config['mongodb']['collections']['conversations']].insert_one({
                'type': 'session_start',
                'session_id': self.session_id,
                'timestamp': datetime.datetime.now(),
                'description': 'EchoMotion AI Session'
            })
            
            logger.info(f"MongoDB connected successfully. Session ID: {self.session_id}")
            
            # Initialize conversation context for the current session only
            self.conversation_history = []
            self.previous_locations = {}
            self.load_conversation_context()
            
            # Count previous interactions in this session only
            session_interactions = self.conversations_collection.count_documents({
                'type': 'interaction',
                'session_id': self.session_id
            })
            
            logger.info(f"Memory system initialized with {session_interactions} interactions in current session")
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            self.mongo_client = None
            self.db = None
            self.conversations_collection = None
            self.session_id = None
            self.conversation_history = []
            self.previous_locations = {}
            
    def _cleanup_old_sessions(self):
        """Mark any incomplete sessions as closed"""
        if self.conversations_collection is None:
            return
            
        try:
            # Find sessions that were started but not explicitly ended
            open_sessions = self.conversations_collection.find({
                'type': 'session_start',
                'closed': {'$ne': True}
            })
            
            # Mark them as closed to prevent cross-session contamination
            for session in open_sessions:
                self.conversations_collection.update_one(
                    {'_id': session['_id']},
                    {'$set': {'closed': True, 'closed_at': datetime.datetime.now()}}
                )
                logger.info(f"Cleaned up old session: {session['session_id']}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            
    def close_session(self):
        """Properly close the current session when the program completely exits"""
        if self.conversations_collection is None or self.session_id is None:
            return
            
        try:
            # Only close the session if this is the true program exit
            # We can detect true program exit by checking if we're exiting from the global Python interpreter
            import sys
            if not sys._getframe(1).f_code.co_name.startswith('__'):  # Not a Python internal method like __del__
                logger.info(f"Not closing session {self.session_id} yet - not a true program exit")
                return
                
            # Clear the persistent session ID  
            SceneGraph._persistent_session_id = None
            
            # Mark the current session as closed
            self.conversations_collection.update_one(
                {'type': 'session_start', 'session_id': self.session_id},
                {'$set': {'closed': True, 'closed_at': datetime.datetime.now()}}
            )
            logger.info(f"Session {self.session_id} closed properly")
            
            # Save final session summary
            interaction_count = self.conversations_collection.count_documents({
                'type': 'interaction', 'session_id': self.session_id
            })
            
            self.conversations_collection.insert_one({
                'type': 'session_end',
                'session_id': self.session_id,
                'timestamp': datetime.datetime.now(),
                'interaction_count': interaction_count,
                'description': f'Session ended with {interaction_count} interactions'
            })
            
        except Exception as e:
            logger.error(f"Error closing session: {e}")
    
    def load_conversation_context(self, limit=10):
        """Load recent conversation context from MongoDB"""
        if self.conversations_collection is None:
            return
            
        # Log session activity to show it's being maintained
        logger.info(f"Loading conversation context for session {self.session_id}")
            
        try:
            # Get the most recent interactions for the current session only
            recent_interactions = self.conversations_collection.find(
                {'type': 'interaction', 'session_id': self.session_id},
                sort=[('timestamp', pymongo.DESCENDING)],
                limit=limit
            )
            
            # Process and store in conversation history
            self.conversation_history = []
            for interaction in recent_interactions:
                # Store simplified version for context
                context_item = {
                    'timestamp': interaction['timestamp'],
                    'text': interaction['input']['text'] if 'input' in interaction and 'text' in interaction['input'] else "",
                    'target_object': interaction['processing']['target_object'] if 'processing' in interaction and 'target_object' in interaction['processing'] else "",
                    'anchor_objects': interaction['processing']['anchor_objects'] if 'processing' in interaction and 'anchor_objects' in interaction['processing'] else [],
                    'location': interaction['response']['location']['center'] if 'response' in interaction and 'location' in interaction['response'] and 'center' in interaction['response']['location'] else None
                }
                
                # Store in history
                self.conversation_history.append(context_item)
                
                # Store locations by object name for easy retrieval
                if context_item['target_object'] and context_item['location']:
                    self.previous_locations[context_item['target_object']] = context_item['location']
                    
            # Reverse to have oldest first for context building
            self.conversation_history.reverse()
            
            logger.info(f"Loaded {len(self.conversation_history)} previous interactions for session {self.session_id}")
                
        except Exception as e:
            logger.error(f"Error loading conversation context: {e}")
            
    def get_location_context(self, text):
        """This method is kept for compatibility but no longer uses hard-coded rules
        Context handling is now delegated to the LLM through conversation history"""
        # We'll let the LLM (OpenAI) handle context understanding through conversation history
        # Just return None, None to allow the normal flow to proceed with conversation context
        return None, None
        
    def get_conversation_context_for_llm(self, limit=5):
        """Format conversation history for use in LLM prompts with enhanced context"""
        if not self.conversation_history:
            return ""
            
        # Reload conversation context to ensure we have the most up-to-date information
        self.load_conversation_context()
            
        # Use more recent conversations for richer context (increased from 3 to 5)
        recent_history = self.conversation_history[-limit:] if len(self.conversation_history) > limit else self.conversation_history
        
        # Create a more comprehensive context for the LLM to better understand references
        context_str = """Please consider the following conversation history when interpreting the current request. 
        Pay special attention to object references, spatial relationships, and any references to 'previous' or 'last' locations.
        
PREVIOUS CONVERSATION HISTORY:\n"""
        
        # Include all object locations seen so far for better context
        location_memory = {}
        for interaction in self.conversation_history:
            if interaction['target_object'] and interaction['location']:
                loc_str = [f"{value:.2f}" for value in interaction['location']] if isinstance(interaction['location'], list) else str(interaction['location'])
                location_memory[interaction['target_object']] = loc_str
        
        # Add chronological interactions with richer detail
        for i, interaction in enumerate(recent_history, 1):
            # Format timestamps for better readability
            timestamp = ""
            if 'timestamp' in interaction:
                if hasattr(interaction['timestamp'], 'strftime'):
                    timestamp = interaction['timestamp'].strftime("%H:%M:%S")
                else:
                    timestamp = str(interaction['timestamp'])
            
            context_str += f"{i}. [Turn {i}] User said: '{interaction['text']}'\n"
            context_str += f"   → System identified target object: {interaction['target_object']}\n"
            
            if interaction['anchor_objects']:
                context_str += f"   → In relation to: {', '.join(interaction['anchor_objects'])}\n"
            
            if interaction['location']:
                loc_str = [f"{value:.2f}" for value in interaction['location']] if isinstance(interaction['location'], list) else str(interaction['location'])
                context_str += f"   → Object was located at coordinates: {loc_str}\n"
            
            context_str += "\n"
        
        # Include a summary of known object locations
        if location_memory:
            context_str += "KNOWN OBJECT LOCATIONS:\n"
            for obj, loc in location_memory.items():
                context_str += f"- {obj}: {loc}\n"
            context_str += "\n"
        
        return context_str

    def safe_convert_for_mongodb(self, obj):
        """Safely convert objects for MongoDB storage with improved numpy handling and robust type checks"""
        try:
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, np.ndarray):
                # Only call .tolist() on true np.ndarray
                try:
                    return obj.tolist()
                except Exception as e:
                    logger.warning(f"Could not convert np.ndarray to list: {e}, type: {type(obj)}, value: {obj}")
                    # Fallback: try flattening or string
                    try:
                        return obj.flatten().tolist()
                    except Exception as e2:
                        logger.warning(f"Fallback flatten failed: {e2}, using string representation")
                        return str(obj)
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                # Handle numpy scalars - use item() method safely
                try:
                    return obj.item()
                except Exception as e:
                    logger.warning(f"Could not convert numpy scalar using item(): {e}, type: {type(obj)}, value: {obj}")
                    try:
                        if isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.bool_):
                            return bool(obj)
                        else:
                            return str(obj)
                    except Exception as e2:
                        logger.warning(f"Direct conversion also failed: {e2}, using string representation")
                        return str(obj)
            elif isinstance(obj, np.generic):
                # Handle other numpy types
                try:
                    if hasattr(obj, 'item'):
                        return obj.item()
                    else:
                        return str(obj)
                except Exception as e:
                    logger.warning(f"Could not convert np.generic: {e}, type: {type(obj)}, value: {obj}")
                    return str(obj)
            elif hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist', None)):
                # Fallback for objects with tolist but not ndarray (e.g., masked arrays)
                try:
                    return obj.tolist()
                except Exception as e:
                    logger.warning(f"Could not convert object with tolist(): {e}, type: {type(obj)}, value: {obj}")
                    return str(obj)
            elif isinstance(obj, (list, tuple)):
                return [self.safe_convert_for_mongodb(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: self.safe_convert_for_mongodb(v) for k, v in obj.items()}
            elif hasattr(obj, '__dict__'):
                # Handle objects with attributes
                return self.safe_convert_for_mongodb(obj.__dict__)
            else:
                # Fallback to string representation
                return str(obj)
        except Exception as e:
            logger.warning(f"Error converting object {type(obj)} with value {obj}: {e}, using string representation")
            return str(obj)

    def store_conversation(self, text_input, target_object, anchor_objects, response_objects, response_relations, pred_center):
        """Store conversation data in MongoDB with improved error handling"""

        if self.conversations_collection is None:
            logger.warning("MongoDB not connected. Cannot store conversation.")
            return

        try:


            # Convert all data safely before building the document
            safe_text_input = str(text_input) if text_input is not None else ""
            safe_target_object = str(target_object) if target_object is not None else ""
            safe_anchor_objects = [str(obj) for obj in (anchor_objects if anchor_objects else [])]
            
            # Handle response objects and relations
            safe_response_objects = self.safe_convert_for_mongodb(response_objects) if response_objects is not None else None
            safe_response_relations = self.safe_convert_for_mongodb(response_relations) if response_relations is not None else None

            logger.info(f"Storing conversation with target object: {safe_target_object}, anchor objects: {safe_anchor_objects}")
            
            # Build the conversation data structure with safely converted data
            conversation_data = {
                'type': 'interaction',
                'session_id': self.session_id,
                'timestamp': datetime.datetime.now(),
                'input': {
                    'text': safe_text_input
                },
                'processing': {
                    'target_object': safe_target_object,
                    'anchor_objects': safe_anchor_objects
                },
                'response': {
                    'objects': safe_response_objects,
                    'relations': safe_response_relations,
                }
            }

            # Insert into MongoDB
            result = self.conversations_collection.insert_one(conversation_data)
            logger.info(f"Conversation stored successfully with ID: {result.inserted_id}")
            return result.inserted_id
            
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error args: {e.args}")

            return None

    def inference(self, batch):
        scene_id = batch['meta']['scene_id']
        text = transcribe_audio_to_text()
        print(f'Text is {text}')
        
        if self.mode == 'gt':
            self.load_gt_scene(batch)
        
        # Reload the conversation context from the database to get any new interactions
        # But don't create a new session - we'll use the existing session ID
        self.load_conversation_context()
        
        # Get conversation context to provide to the LLM
        conversation_context = self.get_conversation_context_for_llm()
        
        # Let the LLM handle all context understanding through conversation history
        target_object, anchor_objects, response_objects = \
            self.talker.ask_objects(text, self.id2obj[scene_id], conversation_context)
       
        if len(anchor_objects) == 0:
            pred_center, pred_points = self.get_obj_center(self.id2obj[scene_id], target_object)
                  
            # Store conversation data
            self.store_conversation(
                text_input=text,
                target_object=target_object,
                anchor_objects=anchor_objects,
                response_objects=response_objects,
                response_relations=None,
                pred_center=pred_center
            )
            
            # Convert numpy arrays to lists for the return values
            pred_center_ret = pred_center.tolist() if isinstance(pred_center, np.ndarray) else pred_center
            pred_points_ret = pred_points.tolist() if isinstance(pred_points, np.ndarray) else pred_points
            
            return pred_center_ret, pred_points_ret, response_objects, None, text
        else:
            relations = self.scenegraph_relationship(scene_id, target_object, anchor_objects)
            # Get the most up-to-date conversation context
            conversation_context = self.get_conversation_context_for_llm()
            
            target_name, response_relations = \
                self.talker.ask_relations(text, relations, self.id2obj[scene_id], target_object, anchor_objects, conversation_context)
            pred_center, pred_points = self.get_obj_center(self.id2obj[scene_id], target_name)
            
            # Convert numpy arrays to lists for the return values
            pred_center_ret = pred_center.tolist() if isinstance(pred_center, np.ndarray) else pred_center
            pred_points_ret = pred_points.tolist() if isinstance(pred_points, np.ndarray) else pred_points

            # Store conversation data
            self.store_conversation(
                text_input=text,
                target_object=target_name,
                anchor_objects=anchor_objects,
                response_objects=response_objects,
                response_relations=response_relations,
                pred_center=pred_center
            )
            
            return pred_center_ret, pred_points_ret, response_objects, response_relations, text
   
 
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

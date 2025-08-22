import os
import yaml
from pathlib import Path

import tenacity
from openai import AzureOpenAI
import logging
import requests
import json
import torch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'configurations.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
config = load_config()
LLM_BASE_URL = config['LLM_BASE_URL']
openai_config = config['azure_openai']
class ChatGPTTalker():
    def __init__(self, prompt_type='paper'):
        self.prompt_type = prompt_type
        

        self.client = AzureOpenAI(
            azure_deployment=openai_config['deployment'],
            api_version=openai_config['api_version'],
            api_key=openai_config['api_key'],
            azure_endpoint=openai_config['endpoint'],
        )
        self.model_name = openai_config['model_name']
 
    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def ask_objects_gpt(self, text, all_objects, conversation_context=""):
        object_string = ', '.join(all_objects)
        
        if self.prompt_type == 'paper':
            # Create the message for our FastAPI service
            if conversation_context:
                full_text = f"Conversation History : {conversation_context}\n User Request: {text} \n\nPlease make sure that the target object and anchor object are in the room."
            else:
                full_text = text
            
            # Prepare payload for FastAPI service
            payload = {
                "message": full_text,
                "room_objects": all_objects
            }
            print(f"Conversation Context: {conversation_context}")
            try:
                # Make request to FastAPI service
                response = requests.post(
                    f"{LLM_BASE_URL}/detect",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30  # Add timeout to prevent hanging
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Format response to match EXACT expected format
                    target = result['target']
                    anchor = result['anchor'] if result['anchor'] and result['anchor'].lower() != 'none' else None
                    
                    if anchor:
                        formatted_response = f"target: {target}\nanchor: {anchor}"
                    else:
                        formatted_response = f"target: {target}\nanchor: None"
                    
                    return {
                        'response': formatted_response,
                        'input_text': text,
                        'input_object': object_string,
                        'reasoning': result.get('reasoning', ''),  # Additional field with reasoning
                        'raw_result': result  # Keep raw result for debugging
                    }
                else:
                    # Handle API errors
                    error_msg = f"API Error: {response.status_code} - {response.text}"
                    print(f"Error calling LLM API: {error_msg}")
                    
                    # Return fallback response
                    return {
                        'response': f"target: {all_objects[0] if all_objects else 'unknown'}\nanchor: None",
                        'input_text': text,
                        'input_object': object_string,
                        'error': error_msg
                    }
                    
            except requests.exceptions.RequestException as e:
                # Handle network errors
                error_msg = f"Network Error: {str(e)}"
                print(f"Error calling LLM API: {error_msg}")
                
                # Return fallback response
                return {
                    'response': f"target: {all_objects[0] if all_objects else 'unknown'}\nanchor: None",
                    'input_text': text,
                    'input_object': object_string,
                    'error': error_msg
                }
        else:
            raise NotImplementedError
   
   
    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def ask_relation_gpt(self, text, relations, target_object, anchor_objects, conversation_context=""):
        assert relations != []
        all_relations = [v for k, v in relations.items()]
        relation_string = '; '.join(all_relations)
        anchor_string = ', '.join(anchor_objects)
        if self.prompt_type == 'paper':
            messages = [
                {
                    "role": "system",
                    "content": "You are an assistant that determine the target object name. Given a text description and the relation information in a room, you should determine the target object name that the text description specifies. If you cannot find the answer, just make a guess."},
                {
                    "role": "assistant",
                    "content": """
                        The relations are split by ";". For each relations, the format is:
                            <target object>, <anchor objects>, <relationship>
                        Here are the examples:
                        Assume the relation in the room is: chair 15 is near to table 1; paper 11 is above the sofa; bed 1 is between the tabel 10 and door 1;
                        1. Sit on the chair that is next to the tables. Please answer: target: chair 15
                        2. Lie on the bed that is in the center of the tables and the door. Please answer: target: bed 1
                        3. Walk to the paper that is above the sofa. Please answer: target: paper 11
                    """,
                },
                {
                    "role": "user",
                    "content": f"""
                        The room contains: {relation_string}. 
                        {conversation_context}
                        The text description is: {text}. Previously, you have found that the target object is {target_object} and the anchor objects are {anchor_string}. If you cannot find the answer, just make a guess.
                        Please provide your thinking process along with your answer. Then the answers should be in a new line. There should be only one target object. Please answer in the following format without any explanations: target: <target object>
                    """,
                }
            ]
        else:
            raise NotImplementedError
       
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.5,
            max_tokens=300,
        )
        return {
            'response': response.choices[0].message.content,
            'input_text': text,
            'input_relation': relation_string,
        }
   
    def check_objects(self, response, all_objects):
        lines = response['response'].split('\n')
        flag_target = False
        flag_anchor = False  # Fixed typo: was 'flat_anchor'
        target_object = None
        anchor_objects = []
        
        # Create a copy of all_objects to avoid modifying during iteration
        available_objects = all_objects.copy()
        
        for line in lines:
            line = line.lower()
            if 'target:' in line:
                flag_target = True
                idx = line.find('target:')
                target_object = line[idx+7:].strip()
                if target_object not in available_objects:
                    result = classify(target_object, available_objects)
                    if result[0] is not None:
                        target_object = result[0]
                        available_objects.remove(target_object)
                else:
                    available_objects.remove(target_object)
                    
            if 'anchor:' in line:
                flag_anchor = True
                if 'none' in line:
                    anchor_objects = []
                else:
                    idx = line.find('anchor:')
                    anchor_objects = line[idx+7:].strip().split(',')
                    for i in range(len(anchor_objects)):
                        anchor_objects[i] = anchor_objects[i].strip()  # Remove whitespace
                        if anchor_objects[i] not in available_objects:
                            result = classify(anchor_objects[i], available_objects)
                            if result[0] is not None:
                                anchor_objects[i] = result[0]
                                if anchor_objects[i] in available_objects:
                                    available_objects.remove(anchor_objects[i])
                        else:
                            available_objects.remove(anchor_objects[i])
        
        if not flag_target or not flag_anchor:
            return None, None
        
        return target_object, anchor_objects
    def check_target(self, response, all_objects):
        lines = response["response"].split('\n')
 
        flag = False
        for line in lines:
            line = line.lower()
            if 'target:' in line.lower():
                flag = True
                idx = line.find('target:')
                target_object = line[idx+7:].strip()
                if target_object not in all_objects:
                    target_object, sim = classify(target_object, all_objects)
        if not flag:
            target_object, sim = classify(response['response'], all_objects)
        return target_object
   
    def ask_objects(self, text, obj_dict, conversation_context=""):
        '''
            Output: target_object, anchor_objects, response
        '''
        all_objects = [obj for obj in obj_dict.keys()]
        target_object = None
        while target_object is None:
            response = self.ask_objects_gpt(text, all_objects, conversation_context)
            logger.info(f"\nResponse from GPT: \n{response['response']}")
            target_object, anchor_objects = self.check_objects(response, all_objects)
        return target_object, anchor_objects, response
 
    def ask_relations(self, text, relations, obj_dict, target_object, anchor_objects, conversation_context=""):
        '''
            Output: target_object, response
        '''
        all_objects = []
        for label, lable_objects in obj_dict.items():
            for obj in lable_objects:
                all_objects.append(obj["name"])
       
        response = self.ask_relation_gpt(text, relations, target_object, anchor_objects, conversation_context)
        target_object = self.check_target(response, all_objects)
        return target_object, response
 
 
import clip
import numpy as np

clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu',
                                        jit=False)  # Must set jit=False for training
clip_model = clip_model.float()
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad_(False)
clip_model = clip_model.cuda()
 
# def clip_feat(w):
#     text_token = clip.tokenize(w).cuda()
#     with torch.no_grad():
#         text_feature = clip_model.encode_text(text_token)
#     # Ensure numpy output (1, D) → (D,)
#     return text_feature.cpu().numpy()[0]

 


def clip_feat(w):
    text_token = clip.tokenize(w)
    text_feature = clip_model.encode_text(text_token.cuda()).cpu()
    # Convert to numpy array and detach from computation graph
    return text_feature.detach().numpy()

 
def similarity(phrase1, phrase2):
    v1 = clip_feat(phrase1)[0]
    v2 = clip_feat(phrase2)[0]
    # Ensure both vectors are numpy arrays
    v1 = np.array(v1) if not isinstance(v1, np.ndarray) else v1
    v2 = np.array(v2) if not isinstance(v2, np.ndarray) else v2
    
    # Calculate cosine similarity
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    # Avoid division by zero
    if norms == 0:
        return 0.0
    
    return dot_product / norms

 
def classify(word, classes):
    if not classes:  # Handle empty classes list
        return None, 0.0
    
    # Truncate word to 77 characters (CLIP's max token length)
    truncated_word = word[:77] if len(word) > 77 else word
    
    similarities = []
    for c in classes:
        try:
            sim = similarity(truncated_word, c)
            similarities.append(sim)
        except Exception as e:
            print(f"Warning: Could not compute similarity for '{truncated_word}' and '{c}': {e}")
            similarities.append(0.0)
    
    if not similarities or max(similarities) == 0:
        return classes[0], 0.0  # Return first class as fallback
    
    max_idx = similarities.index(max(similarities))
    return classes[max_idx], max(similarities)
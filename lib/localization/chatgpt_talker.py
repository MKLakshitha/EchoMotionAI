import os
import yaml
from pathlib import Path

import tenacity
from openai import AzureOpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'configurations.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class ChatGPTTalker():
    def __init__(self, prompt_type='paper'):
        self.prompt_type = prompt_type
        
        # Load configuration from YAML file
        config = load_config()
        openai_config = config['azure_openai']
        
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
            messages = [
                {
                    "role": "system",
                    "content": "You are an assistant that helps people find objects in a room. You are given a list of objects in a room together with a text descriptions. You should determine the target object and anchor object in the text description and map it to the objects in the room. If the object is in the room, just pick it. However, if the object cannot be find in the room, you should pick a room object that is the most similar to the target object."},
                {
                    "role": "assistant",
                    "content": """
                        Here are the examples:
                        Assume the room has: table, sofa chair, door, bed, washing machine, toliet.
                        Please note that anchors should be split by ",".
                        1. Walk to the bathroom vanity. Please answer:
                            target: toliet
                            anchor: None
                        2. Sit on the chair that is next to the tables. Please answer:
                            target: sofa chair
                            anchor: table
                        3. Lie on the tables that is in the center of the door and the bed. Please answer:
                            target: table
                            anchor: door, bed
                        4. Stand up from the chair that is next to the tables. Please answer:
                            target: sofa chair
                            anchor: table
                    """,
                },
                {
                    "role": "user",
                    "content": f"""
                        The room contains: {object_string}. 
                        {conversation_context}
                        The text description is: {text}.
                        Please make sure that the target object and anchor object are in the room. If you cannot find the answer, just make a guess.
                        Answer should be in the following format without any explanations: target: <target object>\nanchor: <anchor object>\n
                    """,
                }
            ]
        else:
            raise NotImplementedError
               
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.5,
        )
        return {
            'response': response.choices[0].message.content,
            'input_text': text,
            'input_object': object_string,
        }
   
   
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
        flat_anchor = False
        for line in lines:
            line = line.lower()
            if 'target:' in line:
                flag_target = True
                idx = line.find('target:')
                target_object = line[idx+7:].strip()
                if target_object not in all_objects:
                    target_object, sim = classify(target_object, all_objects)
                all_objects.remove(target_object)
            if 'anchor:' in line:
                flat_anchor = True
                if 'none' in line:
                    anchor_objects = []
                else:
                    idx = line.find('anchor:')
                    anchor_objects = line[idx+7:].strip().split(',')
                    for i in range(len(anchor_objects)):
                        if anchor_objects[i] not in all_objects:
                            anchor_objects[i], sim = classify(anchor_objects[i], all_objects)
                        all_objects.remove(anchor_objects[i])
        if not flag_target or not flat_anchor:
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
            logger.info(f"Response from GPT: {response['response']}")
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
 
def clip_feat(w):
    text_token = clip.tokenize(w)
    text_feature = clip_model.encode_text(text_token.cuda()).cpu()
    # text_feature = clip_model.encode_text(text_token).cpu()
    return text_feature
 
def similarity(phrase1, phrase2):
    v1 = clip_feat(phrase1)[0]
    v2 = clip_feat(phrase2)[0]
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
 
def classify(word, classes):
    similarities = [similarity(word[:77], c) for c in classes]
    return classes[similarities.index(max(similarities))], max(similarities)
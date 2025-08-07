from unsloth import FastLanguageModel, is_bfloat16_supported
import torch


max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)


model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

# System prompt for GRPO reasoning format
SYSTEM_PROMPT = """
You are an assistant that helps people find objects in a room. You are given a list of objects in a room together with a text descriptions. You should determine the target object and anchor object in the text description and map it to the objects in the room. If the object is in the room, just pick it. However, if the object cannot be find in the room, you should pick a room object that is the most similar to the target object.
Extract only the target object name that the user says where the action is needed to be performed.

Respond in the following format:
<reasoning>
Explain how you identified the target object from the user's input.
</reasoning>
<answer>
target: extracted object
anchor: anchor object if needed
</answer>
"""

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


COT_PROMPT = """
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
    """


text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "assistant", "content" : COT_PROMPT},
    {"role" : "user", "content" : "lie on the table that is close to the cabinet"},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

print(output)
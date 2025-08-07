from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import torch
from unsloth import FastLanguageModel
from vllm import SamplingParams
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Object Detection API", version="1.0.0")

# Request/Response models
class ObjectDetectionRequest(BaseModel):
    message: str
    room_objects: Optional[List[str]] 

class ObjectDetectionResponse(BaseModel):
    target: str
    anchor: Optional[str]
    reasoning: str

# Global variables for model
model = None
tokenizer = None
sampling_params = None

# System prompt and examples
SYSTEM_PROMPT = """
You are an assistant that helps people find objects in a room. You are given a list of objects in a room together with a text descriptions. You should determine the target object and anchor object in the text description and map it to the objects in the room. If the object is in the room, just pick it. However, if the object cannot be find in the room, you should pick a room object that is the most similar to the target object.
Extract only the target object name that the user says where the action is needed to be performed.

Respond Exactly in the following format do not change the Below Format:
<reasoning>
Explain how you identified the target object from the user's input.
</reasoning>
<answer>
target: extracted object
anchor: anchor object if needed
</answer>
"""

COT_PROMPT = """
Here are the examples:
Assume the room has: table, sofa chair, door, bed, washing machine, toilet.
Please note that anchors should be split by ",".
1. Walk to the bathroom vanity. Please answer:
    target: toilet
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
If the user has mentioned a previous locations decide it based on the context. 
"""

@app.on_event("startup")
async def load_model():
    """Load the model on startup"""
    global model, tokenizer, sampling_params
    
    try:
        logger.info("Loading model...")
        
        max_seq_length = 1024
        lora_rank = 64
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=0.5,
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
        )
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

def parse_model_output(output: str) -> tuple:
    """Parse the model output to extract reasoning, target, and anchor"""
    try:
        print(f"Raw model output: {output}")
        # Extract reasoning
        reasoning_start = output.find("<reasoning>")
        reasoning_end = output.find("</reasoning>")
        reasoning = ""
        if reasoning_start != -1 and reasoning_end != -1:
            reasoning = output[reasoning_start + 11:reasoning_end].strip()
        
        # Extract answer
        answer_start = output.find("<answer>")
        answer_end = output.find("</answer>")
        answer = ""
        if answer_start != -1 and answer_end != -1:
            answer = output[answer_start + 8:answer_end].strip()
        
        # Parse target and anchor from answer
        target = "unknown"
        anchor = None
        
        lines = answer.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("target:"):
                target = line.replace("target:", "").strip()
            elif line.startswith("anchor:"):
                anchor_text = line.replace("anchor:", "").strip()
                if anchor_text and anchor_text.lower() != "none":
                    anchor = anchor_text
        
        return reasoning, target, anchor
        
    except Exception as e:
        logger.error(f"Error parsing model output: {str(e)}")
        return "Error parsing response", "unknown", None

@app.post("/detect", response_model=ObjectDetectionResponse)
async def detect_object(request: ObjectDetectionRequest):
    """Detect target and anchor objects from user message"""
    global model, tokenizer, sampling_params
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create the prompt with room objects context
        room_objects_text = ", ".join(request.room_objects)
        user_prompt = f"Assume the room has: {room_objects_text}.\nUser request: {request.message}"
        
        # Apply chat template
        text = tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": COT_PROMPT},
            {"role": "user", "content": user_prompt},
        ], tokenize=False, add_generation_prompt=True)
        
        # Generate response
        output = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=model.load_lora("grpo_saved_lora"),
        )[0].outputs[0].text
        
        # Parse the output
        reasoning, target, anchor = parse_model_output(output)
        print(f"Reasoning: {reasoning}")
        print(f"Target: {target}")
        print(f"Anchor: {anchor}")
        return ObjectDetectionResponse(
            target=target,
            anchor=anchor,
            reasoning=reasoning
        )
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "POST /detect - Detect objects from user message",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8005,
        reload=False,
        workers=1
    )
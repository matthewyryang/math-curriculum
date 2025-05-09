from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from typing import Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_insights import compute_contrastive_loss
import argparse

app = FastAPI(title="Contrastive Loss API")

class ContrastiveRequest(BaseModel):
    paper1_examples: Union[List[str], str]
    paper2_examples: Union[List[str], str]
    joint_examples: Union[List[str], str]
    no_context_examples: Union[List[str], str]
    insight_used: Union[List[str], str]

class ContrastiveResponse(BaseModel):
    paper1_scores: List[float]
    paper1_scores_avg: List[float]
    paper2_scores: List[float]
    paper2_scores_avg: List[float]
    joint_scores: List[float]
    joint_scores_avg: List[float]
    no_context_scores: List[float]
    no_context_scores_avg: List[float]
    contrastive_loss: List[float]
    contrastive_loss_avg: List[float]

# Global variables to store model and tokenizer
model = None
tokenizer = None

def setup_model(model_name_or_path: str):
    global model, tokenizer
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    return tokenizer, model

@app.post("/compute_contrastive_loss", response_model=ContrastiveResponse)
async def compute_contrastive_loss_endpoint(request: ContrastiveRequest):
    try: 
        if isinstance(request.paper1_examples, str):
            request.paper1_examples = [request.paper1_examples]
        if isinstance(request.paper2_examples, str):
            request.paper2_examples = [request.paper2_examples]
        if isinstance(request.joint_examples, str):
            request.joint_examples = [request.joint_examples]
        if isinstance(request.no_context_examples, str):
            request.no_context_examples = [request.no_context_examples]
        if isinstance(request.insight_used, str):
            request.insight_used = [request.insight_used]
            
        # Compute contrastive loss
        paper1_scores, paper1_scores_avg, paper2_scores, paper2_scores_avg, joint_scores, joint_scores_avg, no_context_scores, no_context_scores_avg, contrastive_loss, contrastive_loss_avg = compute_contrastive_loss(
            request.paper1_examples,
            request.paper2_examples,
            request.joint_examples,
            request.no_context_examples,
            request.insight_used,
            model,
            tokenizer
        )
        
        # Convert tensors to lists
        return ContrastiveResponse(
            paper1_scores=paper1_scores.cpu().numpy().tolist(),
            paper1_scores_avg=paper1_scores_avg.cpu().numpy().tolist(),
            paper2_scores=paper2_scores.cpu().numpy().tolist(),
            paper2_scores_avg=paper2_scores_avg.cpu().numpy().tolist(),
            joint_scores=joint_scores.cpu().numpy().tolist(),
            joint_scores_avg=joint_scores_avg.cpu().numpy().tolist(),
            no_context_scores=no_context_scores.cpu().numpy().tolist(),
            no_context_scores_avg=no_context_scores_avg.cpu().numpy().tolist(),
            contrastive_loss=contrastive_loss.cpu().numpy().tolist(),
            contrastive_loss_avg=contrastive_loss_avg.cpu().numpy().tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-14B")
    args = parser.parse_args()
    setup_model(args.model_name_or_path)
    uvicorn.run(app, host=args.host, port=args.port) 
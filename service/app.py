# app.py
import os
import time
import asyncio
import traceback
from typing import Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# -------------------------
# Config from env
# -------------------------
MODEL_REPO = os.getenv("MODEL_REPO", "keshav1926s/phi3-lora-adapter")  # your HF repo id (adapter or merged)
HF_TOKEN = os.getenv("HF_TOKEN")  # required for private repos
BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/phi-3-mini-4k-instruct")  # used if MODEL_REPO is adapter-only
USE_4BIT = os.getenv("USE_4BIT", "true").lower() in ("1", "true", "yes")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS", "256"))

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="Inference Service")

# model globals
model = None
tokenizer = None
model_lock = asyncio.Lock()
_ready = False

# request/response schema
class InferRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = MAX_NEW_TOKENS_DEFAULT
    temperature: Optional[float] = 0.0
    do_sample: Optional[bool] = False
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.0
    # any additional kwargs passed to model.generate
    generation_kwargs: Optional[Dict[str, Any]] = {}

class InferResponse(BaseModel):
    text: str
    elapsed: float
    meta: Dict[str, Any]

# -------------------------
# Model loader
# -------------------------
def _log(msg: str):
    print(f"[inference_service] {msg}", flush=True)

def _apply_compatibility_flags(loaded_model):
    """
    Disable KV cache and set eager attention implementation to avoid DynamicCache API
    mismatches (e.g., missing get_usable_length / seen_tokens).
    """
    try:
        # primary flag
        if hasattr(loaded_model, "config"):
            loaded_model.config.use_cache = False
            # set eager attention implementation if available
            try:
                loaded_model.config.attn_implementation = "eager"
            except Exception:
                pass
        # if it's a PEFT wrapper, also try to set on base_model
        try:
            base = getattr(loaded_model, "base_model", None)
            if base is not None and hasattr(base, "config"):
                base.config.use_cache = False
                try:
                    base.config.attn_implementation = "eager"
                except Exception:
                    pass
        except Exception:
            pass
        # ensure model in eval mode
        try:
            loaded_model.eval()
        except Exception:
            pass
        _log("Applied compatibility flags: use_cache=False, attn_implementation='eager' (where supported).")
    except Exception as e:
        _log(f"Warning: failed to apply compatibility flags: {e}")

def load_model_and_tokenizer():
    """
    Attempt to load MODEL_REPO as a merged full model first.
    If that fails, assume MODEL_REPO is an adapter and:
      - load BASE_MODEL (in 4-bit if USE_4BIT)
      - attach adapter via PeftModel.from_pretrained(MODEL_REPO)
    Requires HF_TOKEN for private repos.
    """
    global model, tokenizer, _ready

    _log(f"Starting model load. MODEL_REPO={MODEL_REPO}, BASE_MODEL={BASE_MODEL}, USE_4BIT={USE_4BIT}, DEVICE={DEVICE}")
    auth_kwargs = {}
    if HF_TOKEN:
        auth_kwargs["use_auth_token"] = HF_TOKEN

    # try: load repo as full model
    try:
        _log("Trying to load MODEL_REPO as a full model from the Hub...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            device_map="auto" if DEVICE == "cuda" else None,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True,
            **auth_kwargs,
        )
        # apply compatibility flags right after loading
        _apply_compatibility_flags(model)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True, **auth_kwargs)
        _log("Loaded merged full model successfully from repo.")
        _ready = True
        return
    except Exception as e_full:
        _log("Loading as merged model failed (this may be expected if repo contains only adapter).")
        _log(f"Full-model load exception: {e_full}")
        _log("Attempting adapter flow (load base model then PeftModel.from_pretrained)...")

    # adapter flow
    try:
        tokenizer = None
        # try to load tokenizer from adapter repo (if present), else from base model
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True, **auth_kwargs)
            _log("Loaded tokenizer from adapter repo.")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, **auth_kwargs)
            _log("Adapter repo had no tokenizer; loaded tokenizer from base model.")

        # bitsandbytes config (4-bit) if requested
        bnb_config = None
        if USE_4BIT:
            # compute_dtype set to float16 for most GPUs (V100)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            _log("Configured BitsAndBytesConfig for 4-bit loading.")

        # load base model (possibly quantized)
        _log(f"Loading base model: {BASE_MODEL} (may take a while)...")
        if USE_4BIT and bnb_config is not None:
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True,
                **({"torch_dtype": torch.float16} if DEVICE == "cuda" else {}),
                **auth_kwargs,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="auto" if DEVICE == "cuda" else None,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                trust_remote_code=True,
                **auth_kwargs,
            )

        _log("Attaching adapter from MODEL_REPO via PEFT...")
        model_peft = PeftModel.from_pretrained(base, MODEL_REPO, device_map="auto" if DEVICE == "cuda" else None, **auth_kwargs)
        model = model_peft

        # apply compatibility flags after PEFT attachment
        _apply_compatibility_flags(model)

        _log("Peft adapter attached successfully.")
        _ready = True
        return
    except Exception as e_adapter:
        _log("Adapter loading flow failed.")
        _log(traceback.format_exc())
        raise RuntimeError("Failed to load model (merged or adapter flows). See logs.") from e_adapter


# Launch model load at startup in background thread so server can respond to /healthz quickly
@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    # run blocking load in default executor so uvicorn event loop isn't blocked
    loop.run_in_executor(None, load_model_and_tokenizer)
    _log("Model loading scheduled in background.")

# -------------------------
# Health endpoints
# -------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    return {"ready": bool(_ready)}

# -------------------------
# Inference endpoint
# -------------------------
@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    if not _ready:
        raise HTTPException(status_code=503, detail="Model not ready yet. Check /readyz")
    prompt = req.prompt
    gen_kwargs = dict(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        do_sample=req.do_sample,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    # merge any additional generation kwargs
    if req.generation_kwargs:
        gen_kwargs.update(req.generation_kwargs)

    t0 = time.time()
    async with model_lock:
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # ensure attention_mask present
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]).to(model.device)

            # generate (run in non-blocking if GPU-bound)
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            _log("Generation error: " + str(e))
            _log(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    elapsed = time.time() - t0
    return InferResponse(text=text, elapsed=elapsed, meta={"model_repo": MODEL_REPO, "device": str(model.device), "gen_kwargs": gen_kwargs})


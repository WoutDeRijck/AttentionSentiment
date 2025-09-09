"""
Attention model module using Google Gemma-3-1B-IT from Hugging Face.

This module implements attention extraction from the Gemma model for sentiment analysis
visualization.
"""

import torch
import numpy as np
from transformers import pipeline

# Toggle to enable/disable debug prints
DEBUG_ATTENTION_LOGS = True
 

prompt = """
You are an email classifier. Respond with EXACTLY one of the label identifiers below, with no extra words.

I have the following email:

{text}

Classify the email into ONE of these categories (respond with the identifier in parentheses only):

Attention needed(Attention_needed)
ONLY select when the customer explicitly uses escalation language:
- Contains words: "urgent", "escalate", "emergency", "immediate attention required", "customers affected"
- Customer demands management involvement
- Uses phrases like "need to escalate this", "urgent request", "emergency priority"
DO NOT select based on:
- Technical severity words (crash, down, failure, critical error, slow, broken)
- Impact descriptions (systems affected, outages, performance issues)
Technical problems (even severe ones like "engine crash", "system down", "critical failure") should be "Default" unless customer explicitly requests escalation.

Default(Default)
Use for all standard technical issues and requests, including:
- System crashes, failures, outages, performance problems
- Any severity level (Minor/Major) without explicit escalation language
- Technical emergencies described without escalation requests
Select this unless customer explicitly uses escalation language like "urgent", "escalate", "emergency priority".

Escalate(Escalate)
Really rare, should be escalated to management. Highest level of escalation possible. Use only when the customer or we explicitly escalate or request escalation at the highest level.

Respond with only one identifier: Attention_needed, Default, or Escalate. Say nothing else.
"""


# Global variables to cache the pipeline, model and tokenizer
_pipeline = None
_model = None
_tokenizer = None

def load_model():
    """
    Load the Gemma-3-1B-IT model and tokenizer via pipeline.
    
    Returns:
        tuple: (model, tokenizer)
    """
    global _pipeline, _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        try:
            # Create the pipeline first with eager attention for output_attentions support
            # Robust device + dtype selection
            if torch.cuda.is_available():
                target_device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                target_device = "mps"
            else:
                target_device = "cpu"

            if DEBUG_ATTENTION_LOGS:
                print(f"[load_model] Selecting pipeline on device={target_device}")

            _pipeline = pipeline(
                "text-generation", 
                # model="google/gemma-3-1b-it",
                model="meta-llama/Llama-3.1-8B-Instruct",
                device=target_device,
                model_kwargs={"attn_implementation": "eager"}  # Force eager attention for output_attentions
            )

            # _pipeline = pipeline(
            #     "text-classification",
            #     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            #     device=target_device,
            #     model_kwargs={"attn_implementation": "eager"}  # Force eager attention for output_attentions
            # )
            
            # Extract model and tokenizer from pipeline
            _model = _pipeline.model
            _tokenizer = _pipeline.tokenizer
            try:
                _model.eval()
                # Ensure safe dtype on CPU
                if target_device == "cpu":
                    _model.to(torch.float32)
            except Exception:
                pass
            
            # Add padding token if it doesn't exist
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            # Ensure model forwards can return attentions
            try:
                _model.config.output_attentions = True
            except Exception:
                pass
            
            print("✅ Successfully loaded model via pipeline:", getattr(_pipeline, 'task', None), "| model:", _model.__class__.__name__)
            
        except Exception as e:
            print(f"❌ Error loading Gemma model: {e}")
            # Fallback to a smaller model if Gemma fails
            try:
                # Reuse the same robust device setup for fallback
                if torch.cuda.is_available():
                    target_device = "cuda"
                elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"
                _pipeline = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    device=target_device,
                    model_kwargs={"attn_implementation": "eager"}  # Force eager attention for fallback too
                )
                _model = _pipeline.model
                _tokenizer = _pipeline.tokenizer
                try:
                    _model.eval()
                    if target_device == "cpu":
                        _model.to(torch.float32)
                except Exception:
                    pass
                
                if _tokenizer.pad_token is None:
                    _tokenizer.pad_token = _tokenizer.eos_token
                    
                print("✅ Fallback: Successfully loaded distilgpt2")
            except Exception as fallback_e:
                print(f"❌ Fallback also failed: {fallback_e}")
                raise fallback_e
    
    return _model, _tokenizer


def _take_first_word(text: str) -> str:
    """Return the first word-like token from text (letters/numbers), stripped of punctuation."""
    if not text:
        return ""
    parts = text.strip().split()
    if not parts:
        return ""
    token = parts[0]
    try:
        import re
        token = re.sub(r"^[^\w]+|[^\w]+$", "", token)
    except Exception:
        pass
    return token


def tokenize_text(text, tokenizer):
    """
    Tokenize the input text using the model's tokenizer.
    
    Args:
        text (str): Input text to tokenize
        tokenizer: The model's tokenizer
        
    Returns:
        dict: Tokenized inputs with input_ids, attention_mask, etc.
    """
    # Tokenize with proper padding and truncation
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,  # Limit to 512 tokens for memory efficiency
        return_attention_mask=True
    )
    
    return inputs


def _row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    """Row-normalize a square attention matrix so rows sum to 1.
    Args:
        matrix: Tensor of shape [seq_len, seq_len]
    Returns:
        Tensor of shape [seq_len, seq_len]
    """
    eps = matrix.new_tensor(1e-12)
    row_sum = matrix.sum(dim=-1, keepdim=True) + eps
    return matrix / row_sum


def _build_a_tilde(attentions):
    """Build per-layer row-normalized (A + I) after averaging heads.
    Returns list of torch tensors [seq_len, seq_len].
    """
    device = attentions[0].device
    layer_mats = [att[0].mean(dim=0) for att in attentions]
    a_tilde = []
    for A in layer_mats:
        seq_len = A.size(-1)
        A_res = A + torch.eye(seq_len, device=device, dtype=A.dtype)
        A_tilde = _row_normalize(A_res)
        if DEBUG_ATTENTION_LOGS:
            with torch.no_grad():
                try:
                    rs_min = float(A_tilde.sum(dim=-1).min())
                    rs_max = float(A_tilde.sum(dim=-1).max())
                    a_min = float(A_tilde.min())
                    a_max = float(A_tilde.max())
                    print(f"[_build_a_tilde] seq_len={seq_len} row_sum_min={rs_min:.6f} row_sum_max={rs_max:.6f} val_min={a_min:.6f} val_max={a_max:.6f}")
                except Exception:
                    pass
        a_tilde.append(A_tilde)
    return a_tilde


def _mask_and_normalize_attentions(attentions, noise_mask: np.ndarray):
    """
    Zero out attention to and from tokens where noise_mask[i] is True, then row-normalize.
    Args:
        attentions: tuple of [batch, heads, seq, seq]
        noise_mask: numpy array [seq] with True for tokens to mask (punct/stopwords/etc.)
    Returns:
        list of torch.FloatTensor [batch, heads, seq, seq] with masked and normalized rows.
    """
    if noise_mask is None:
        return attentions
    # Convert to torch bool mask on same device as attentions
    device = attentions[0].device
    mask_t = torch.from_numpy(noise_mask.astype(np.bool_)).to(device)
    masked = []
    # match dtype for numerical stability across devices
    eps = attentions[0].new_tensor(1e-12)
    for layer_idx, att in enumerate(attentions):
        # Work in float32 to avoid low-precision degeneracy on CPU/MPS
        A = att.clone().to(dtype=torch.float32)  # [batch, heads, seq, seq]
        if DEBUG_ATTENTION_LOGS:
            with torch.no_grad():
                m, M = float(A.min()), float(A.max())
                print(f"[mask_norm] layer={layer_idx} pre-mask stats min={m:.6f} max={M:.6f}")
        # Zero out to and from masked tokens
        A[:, :, mask_t, :] = 0.0
        A[:, :, :, mask_t] = 0.0
        # Row-normalize per head
        row_sum = A.sum(dim=-1, keepdim=True) + eps
        A = A / row_sum
        if DEBUG_ATTENTION_LOGS:
            with torch.no_grad():
                m, M = float(A.min()), float(A.max())
                rs = A.sum(dim=-1).mean().item()
                print(f"[mask_norm] layer={layer_idx} post-norm stats min={m:.6f} max={M:.6f} avg_row_sum={rs:.6f}")
        masked.append(A)
    return tuple(masked)


def compute_attention_rollout(attentions, decision_index: int = -1, noise_mask: np.ndarray = None) -> np.ndarray:
    """
    Compute attention rollout across all layers.
    Heads → layer: average heads per layer.
    Residuals: A_tilde = row_normalize(A + I).
    Rollout: multiply A_tilde from bottom to top; take decision token's row as scores.

    Args:
        attentions: tuple of length L, each of shape [batch, heads, seq_len, seq_len]

    Returns:
        numpy array of shape [seq_len] with rollout scores for the decision token over inputs.
    """
    # Use the first (and only) item in batch
    if noise_mask is not None:
        attentions = _mask_and_normalize_attentions(attentions, noise_mask)
    # Promote to float32 for stability if needed
    attentions_f32 = tuple(att.to(dtype=torch.float32) for att in attentions)
    a_tilde = _build_a_tilde(attentions_f32)
    if DEBUG_ATTENTION_LOGS:
        print(f"[rollout] num_layers={len(a_tilde)} seq_len={a_tilde[0].shape[-1] if a_tilde else 0}")
    # Rollout from bottom to top
    R = a_tilde[0]
    for layer_idx in range(1, len(a_tilde)):
        R = torch.matmul(R, a_tilde[layer_idx])
        if DEBUG_ATTENTION_LOGS and layer_idx % max(1, len(a_tilde)//4) == 0:
            with torch.no_grad():
                m, M = float(R.min()), float(R.max())
                print(f"[rollout] after layer {layer_idx} stats min={m:.6f} max={M:.6f}")
    # Decision token: typically the last token in decoder-only models
    # Decision token row (e.g., -1 for decoder, 0 for encoder [CLS])
    decision_row = R[decision_index]
    scores = decision_row.detach().cpu().numpy()
    # Min–max normalize scores for stability
    min_s, max_s = scores.min(), scores.max()
    if max_s > min_s:
        scores = (scores - min_s) / (max_s - min_s)
    else:
        scores = np.full_like(scores, 0.5)
    if DEBUG_ATTENTION_LOGS:
        print(f"[rollout] decision_index={decision_index} min={min_s:.6f} max={max_s:.6f} normalized_min={scores.min():.6f} normalized_max={scores.max():.6f}")
    return scores


def compute_attention_layer_average(attentions, decision_index: int = -1, noise_mask: np.ndarray = None) -> np.ndarray:
    """Plain average of raw attention over heads and layers; use decision row.
    Returns numpy array [seq_len] min–max normalized.
    """
    # Average heads per layer on raw attentions (no residual, no row-normalization)
    if noise_mask is not None:
        attentions = _mask_and_normalize_attentions(attentions, noise_mask)
    # Promote to float32 for numerical stability and to avoid unsupported bf16 ops
    attentions_f32 = tuple(att.to(dtype=torch.float32) for att in attentions)
    layer_mats = [att[0].mean(dim=0) for att in attentions_f32]  # list of [seq, seq]
    # Weight later layers more heavily for better decision attribution in decoder LMs
    try:
        L = len(layer_mats)
        weights = torch.linspace(0.5, 1.5, steps=L, device=layer_mats[0].device, dtype=layer_mats[0].dtype)
        weights = weights / weights.sum()
        stacked = torch.stack(layer_mats, dim=0)
        M = (stacked * weights.view(L, 1, 1)).sum(dim=0)
        if DEBUG_ATTENTION_LOGS:
            print(f"[layer_avg] layer_weights first={float(weights[0]):.4f} last={float(weights[-1]):.4f}")
    except Exception:
        M = torch.stack(layer_mats, dim=0).mean(dim=0)  # fallback uniform
    # Decision token row (e.g., -1 for decoder, 0 for encoder [CLS])
    decision_row = M[decision_index].detach().cpu().numpy()
    scores = decision_row
    if DEBUG_ATTENTION_LOGS:
        try:
            print(f"[layer_avg] decision_index={decision_index} row_min={float(scores.min()):.6f} row_max={float(scores.max()):.6f}")
        except Exception:
            pass
    min_s, max_s = scores.min(), scores.max()
    if max_s > min_s:
        scores = (scores - min_s) / (max_s - min_s)
    else:
        scores = np.full_like(scores, 0.5)
    return scores



def extract_attention_scores(model, tokenizer, text, decision_index: int = -1):
    """
    Extract attention scores from the model for the given text.
    
    Args:
        model: The loaded transformer model (from pipeline)
        tokenizer: The model's tokenizer (from pipeline)
        text (str): Input text
        layer (int): Which layer to extract attention from (-1 for last, -2 for second-to-last)
        head (int): Which attention head to use (0-based)
        
    Returns:
        tuple: (offsets, attention_scores)
    """
    # Tokenize the input following the pipeline approach
    inputs = tokenizer(text, return_tensors="pt")
    # Also build offset mapping and tokens in a separate pass
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]  # list of (start, end)
    input_ids = enc.get("input_ids", [])
    if DEBUG_ATTENTION_LOGS:
        try:
            model_type = getattr(model.config, 'model_type', 'unknown')
            device_name = str(next(model.parameters()).device)
            print(f"[extract] model={model.__class__.__name__} type={model_type} device={device_name} tokens={len(input_ids)}")
        except Exception:
            pass
    
    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    # Get model output with attention (following your example pattern)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Extract attention weights
    attentions = outputs.attentions  # Tuple of attention tensors for each layer
    if DEBUG_ATTENTION_LOGS and attentions is not None:
        try:
            num_layers = len(attentions)
            b, h, s1, s2 = attentions[0].shape
            print(f"[extract] attentions: layers={num_layers} batch={b} heads={h} seq={s1}")
        except Exception:
            pass
    
    if attentions is None:
        # For models that ignore output_attentions unless set on config, try one more time
        try:
            model.config.output_attentions = True
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions
        except Exception:
            attentions = None
    if attentions is None:
        raise ValueError("Model did not return attention weights. Ensure attn_implementation='eager' and output_attentions=True are supported.")
    
    # Build noise mask: punctuation, stopwords, whitespace/newlines, and special tokens
    # Token-level decisions using decoded tokens and offsets
    def is_noise(tok_text: str) -> bool:
        t = tok_text.strip()
        if t == "":
            return True
        import re
        # whitespace/newlines
        if re.fullmatch(r"\s+", tok_text) is not None:
            return True
        # punctuation-only
        if re.fullmatch(r"\W+", t) is not None:
            return True
        return False

    # Decode tokens lightly to detect noise; avoid creating huge strings per token
    decoded_tokens = [tokenizer.decode([tid]) if isinstance(tid, int) else tokenizer.decode([tid[0]]) for tid in input_ids]
    # Special tokens often look like [CLS], [SEP], <s>, </s>, etc.
    noise_mask = np.array([
        (dt.startswith('[') and dt.endswith(']')) or (dt.startswith('<') and dt.endswith('>')) or is_noise(dt)
        for dt in decoded_tokens
    ], dtype=np.bool_)
    if DEBUG_ATTENTION_LOGS:
        try:
            masked_cnt = int(noise_mask.sum())
            print(f"[extract] initial noise mask active tokens={len(noise_mask)-masked_cnt}/{len(noise_mask)} masked={masked_cnt}")
            preview = [dt.replace("\n","\\n") for dt in decoded_tokens[:30]]
            print("[extract] first 30 tokens:", " | ".join(preview))
        except Exception:
            pass
    # Never mask the decision token and ensure we keep useful tokens
    if 0 <= (decision_index if decision_index >= 0 else len(decoded_tokens) + decision_index) < len(decoded_tokens):
        idx = decision_index if decision_index >= 0 else len(decoded_tokens) + decision_index
        noise_mask[idx] = False
    # If masking removes almost everything, disable masking to avoid degenerate 0.5 outputs
    if (np.sum(~noise_mask) < max(2, int(0.1 * len(noise_mask)))):
        noise_mask = None
        if DEBUG_ATTENTION_LOGS:
            print("[extract] disabled noise mask due to excessive masking")

    # Compute three variants with masking
    avg_scores = compute_attention_layer_average(attentions, decision_index=decision_index, noise_mask=noise_mask)
    rollout_scores = compute_attention_rollout(attentions, decision_index=decision_index, noise_mask=noise_mask)
    if DEBUG_ATTENTION_LOGS:
        try:
            print(f"[extract] scores stats avg(min={float(avg_scores.min()):.4f} max={float(avg_scores.max()):.4f}) rollout(min={float(rollout_scores.min()):.4f} max={float(rollout_scores.max()):.4f})")
        except Exception:
            pass
    
    # Return raw per-token scores aligned with offsets (no filtering)
    return offsets, {
        'layer_average': [float(x) for x in avg_scores.tolist()],
        'rollout': [float(x) for x in rollout_scores.tolist()],
    }


def extract_generated_label_attention(model, tokenizer, formatted_prompt: str, email_start: int, email_end: int):
    """
    For decoder-only models: run 1-step generation with attentions and use the
    generated token's attention row to score the context tokens.

    Returns (offsets, scores_dict) where offsets are for the input tokens only.
    """
    # Tokenize twice to get tensors and offsets
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    enc = tokenizer(formatted_prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc.get("offset_mapping", [])
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    # Ensure we have a pad token id to avoid warnings
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
            pad_id = tokenizer.eos_token_id
        except Exception:
            pass
    # Generate a single token with attentions
    gen_kwargs = {
        'max_new_tokens': 1,
        'do_sample': False,
        'output_attentions': True,
        'return_dict_in_generate': True,
    }
    if pad_id is not None:
        gen_kwargs['pad_token_id'] = pad_id
    with torch.no_grad():
        gen_out = model.generate(**inputs, **gen_kwargs)
    # Debug: decode generated label token
    try:
        last_tok_id = int(gen_out.sequences[0, -1].item())
        label_tok_text = tokenizer.decode([last_tok_id])
        if DEBUG_ATTENTION_LOGS:
            print(f"[generated_label] id={last_tok_id} text='{label_tok_text}'")
    except Exception:
        label_tok_text = ''
    # gen_out.attentions: list length T (1). Each item: tuple of layers with [b,h,seq,seq]
    step_atts = tuple(att.to(dtype=torch.float32) for att in gen_out.attentions[0])
    # Use our existing computations with decision_index = -1 (the generated position)
    avg_scores = compute_attention_layer_average(step_atts, decision_index=-1, noise_mask=None)
    rollout_scores = compute_attention_rollout(step_atts, decision_index=-1, noise_mask=None)
    # Select email token indices from offsets
    token_indices = [i for i, (a, b) in enumerate(offsets) if not (b <= email_start or a >= email_end)]
    if not token_indices:
        token_indices = list(range(len(offsets)))
    # Build decoded token texts for selected tokens
    try:
        ids_all = enc.get('input_ids', [])
        toks_all = [tokenizer.decode([tid]) if isinstance(tid, int) else '' for tid in ids_all]
        toks_sel = [toks_all[i] for i in token_indices]
    except Exception:
        toks_sel = []
    def take(arr):
        return [arr[i] for i in token_indices]
    scores_dict = {
        'layer_average': [float(x) for x in take(avg_scores)],
        'rollout': [float(x) for x in take(rollout_scores)],
        '_tokens': toks_sel,
        '_generated_label_token': label_tok_text,
    }
    # Clip offsets to email span for rendering
    clipped_offsets = [(max(a - email_start, 0), max(min(b, email_end) - email_start, 0)) for i, (a, b) in enumerate(offsets) if i in token_indices]
    return clipped_offsets, scores_dict


def extract_forced_label_attention(model, tokenizer, formatted_prompt: str, label_text: str, email_start: int, email_end: int):
    """
    Decoder-only: append the chosen label (e.g., " Positive") to the prompt, run a
    single forward pass with output_attentions, and use the appended label token's
    attention row to score the original prompt tokens only.
    """
    device = next(model.parameters()).device
    # Offsets for prompt-only tokens
    enc_prompt = tokenizer(formatted_prompt, return_offsets_mapping=True, add_special_tokens=True)
    prompt_offsets = enc_prompt.get("offset_mapping", [])
    num_prompt_tokens = len(prompt_offsets)
    # Build combined input with label appended (ensure leading space)
    label_str = (" " + label_text.strip()) if not label_text.startswith(" ") else label_text
    combined_text = formatted_prompt + label_str
    inputs = tokenizer(combined_text, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = tuple(att.to(dtype=torch.float32) for att in outputs.attentions)
    # Aggregate per standard method
    avg_scores_full = compute_attention_layer_average(attentions, decision_index=-1, noise_mask=None)
    rollout_scores_full = compute_attention_rollout(attentions, decision_index=-1, noise_mask=None)
    # Keep only the columns corresponding to the original prompt tokens
    avg_scores = avg_scores_full[:num_prompt_tokens]
    rollout_scores = rollout_scores_full[:num_prompt_tokens]
    # Select prompt token indices overlapping the email substring
    token_indices = [i for i, (a, b) in enumerate(prompt_offsets) if not (b <= email_start or a >= email_end)]
    if not token_indices:
        token_indices = list(range(len(prompt_offsets)))
    def take(arr):
        return [arr[i] for i in token_indices]
    scores_dict = {
        'layer_average': [float(x) for x in take(avg_scores)],
        'rollout': [float(x) for x in take(rollout_scores)],
        '_generated_label_token': label_text,
    }
    clipped_offsets = [(max(a - email_start, 0), max(min(b, email_end) - email_start, 0)) for i, (a, b) in enumerate(prompt_offsets) if i in token_indices]
    if DEBUG_ATTENTION_LOGS:
        try:
            print(f"[forced_label] using label='{label_text}' and slicing to prompt tokens: {num_prompt_tokens}")
        except Exception:
            pass
    return clipped_offsets, scores_dict


def align_tokens_with_words(text, tokens, attention_scores):
    """
    Align subword tokens back to original words and aggregate attention scores.
    
    Args:
        text (str): Original input text
        tokens (list): List of subword tokens
        attention_scores (list): Attention scores for each token
        
    Returns:
        tuple: (words, word_attention_scores)
    """
    words = text.split()
    
    if len(tokens) == 0 or len(attention_scores) == 0:
        return words, [0.5] * len(words)
    
    # Simple span aggregation: map subword scores to word spans by sum, then min–max normalize
    # Note: Without true token-to-word mapping, approximate equally splitting tokens
    word_scores = []
    tokens_per_word = max(1, len(tokens) // len(words))
    for i, _ in enumerate(words):
        start_idx = i * tokens_per_word
        end_idx = min(start_idx + tokens_per_word, len(attention_scores))
        if start_idx < len(attention_scores):
            span_scores = attention_scores[start_idx:end_idx]
            word_scores.append(float(np.sum(span_scores)))
        else:
            word_scores.append(0.0)
    # Min–max normalize
    if len(word_scores) > 0:
        min_s = float(np.min(word_scores))
        max_s = float(np.max(word_scores))
        if max_s > min_s:
            word_scores = [(s - min_s) / (max_s - min_s) for s in word_scores]
        else:
            word_scores = [0.5 for _ in word_scores]
    return words, word_scores


def predict_sentiment_with_distribution(text):
    """
    General classification: compute probabilities over labels
    [Attention_needed, Default, Escalate] using a decoder LM.
    Returns (predicted_label, probability_dict).
    """
    try:
        # Use the pipeline for text generation
        global _pipeline
        if _pipeline is None:
            load_model()  # This will initialize the pipeline
        
        # Build inputs depending on pipeline task
        model = _pipeline.model
        tokenizer = _pipeline.tokenizer
        device = next(model.parameters()).device

        # Decoder LM: use prompt and last-token logits over our three labels
        formatted_prompt = prompt.format(text=text)
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

        labels = ["Attention_needed", "Default", "Escalate"]
        def first_id(label: str):
            ids = tokenizer.encode(label, add_special_tokens=False)
            return ids[0] if ids else None
        ids = [first_id(lbl) for lbl in labels]
        # If any id is None, assign -inf so it won't win
        label_logits = [logits[i].item() if i is not None else -float('inf') for i in ids]
        label_logits_t = torch.tensor(label_logits, device=logits.device)
        probabilities = torch.softmax(label_logits_t, dim=0)
        prob_dict = {labels[i]: float(probabilities[i].item()) for i in range(len(labels))}
        predicted_label = max(prob_dict, key=prob_dict.get)
        # Also provide a raw generation for UI debugging
        try:
            gen = _pipeline(formatted_prompt, max_new_tokens=8, do_sample=False)
            raw_out = gen[0].get('generated_text', '')
            raw_output = raw_out[len(formatted_prompt):].strip() if raw_out.startswith(formatted_prompt) else raw_out
            raw_output = _take_first_word(raw_output)
        except Exception:
            raw_output = ''
        return predicted_label, {**prob_dict, '_raw_output': raw_output, '_prompt': formatted_prompt}
        
    except Exception as e:
        print(f"Error in classification prediction with distribution: {e}")
        # Fallback to simple generation
        try:
            formatted_prompt_local = prompt.format(text=text)
            response = _pipeline(formatted_prompt_local, max_new_tokens=10, do_sample=False)
            generated_text = response[0]['generated_text']
            sentiment_prediction = generated_text[len(formatted_prompt_local):].strip()
            # Take only the first line and then the first word
            first_line = sentiment_prediction.split('\n', 1)[0].strip()
            sentiment_first = _take_first_word(first_line)

            s_low = sentiment_first.lower()
            if 'escalate' in s_low:
                predicted = 'Escalate'
            elif 'attention' in s_low:
                predicted = 'Attention_needed'
            elif 'default' in s_low:
                predicted = 'Default'
            else:
                predicted = 'Default'

            # Return skewed probabilities as fallback
            prob_dict = {'Attention_needed': 0.15, 'Default': 0.7, 'Escalate': 0.15}
            prob_dict[predicted] = 0.7
            remaining = 0.15
            for key in prob_dict:
                if key != predicted:
                    prob_dict[key] = remaining
                    
            return predicted, {**prob_dict, '_raw_output': sentiment_first, '_prompt': formatted_prompt_local}
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return 'Default', {'Attention_needed': 0.2, 'Default': 0.6, 'Escalate': 0.2, '_raw_output': '', '_prompt': ''}


def predict_sentiment(text):
    """
    Backward-compat API: returns the predicted label string.
    """
    label, _ = predict_sentiment_with_distribution(text)
    return label


def get_attention(text):
    """
    Get attention scores for the input text using Gemma-3-1B-IT and predict sentiment.
    
    Args:
        text (str): The input email text to analyze
        
    Returns:
        dict: A dictionary containing:
            - 'attention_scores': List of attention scores (one per word)
            - 'sentiment': Sentiment prediction (Positive, Negative, Neutral)
            - 'probabilities': Dictionary with probabilities for each sentiment
            - 'confidence': Confidence score (max probability)
    """
    try:
        # Load model and tokenizer
        model, tokenizer = load_model()
        
        # Always treat as decoder-only flow for attention visualization
        decision_index = -1

        # Predict sentiment with probabilities using the prompt
        sentiment_prediction, probabilities = predict_sentiment_with_distribution(text)
        
        # For attention extraction, we'll use the formatted prompt to see what the model focuses on
        formatted_prompt = prompt.format(text=text)

        # Pre-tokenize to determine the email span and prefer decision token at last email token for decoder models
        pre_enc = tokenizer(formatted_prompt, return_offsets_mapping=True, add_special_tokens=True)
        pre_offsets = pre_enc.get("offset_mapping", [])
        start_marker = "I have the following email:"
        end_marker = "Categorize the email"
        s = formatted_prompt.find(start_marker)
        e = formatted_prompt.find(end_marker, s if s >= 0 else 0)
        email_start = (s + len(start_marker)) if s >= 0 else 0
        email_end = e if e >= 0 else len(formatted_prompt)
        pre_token_indices = [i for i, (a, b) in enumerate(pre_offsets) if not (b <= email_start or a >= email_end)]
        if pre_token_indices:
            # Prefer the last non-whitespace, non-punctuation, non-special token in the email span
            try:
                import re
                pre_input_ids = pre_enc.get("input_ids", [])
                chosen = None
                for idx in reversed(pre_token_indices):
                    tid = pre_input_ids[idx] if isinstance(pre_input_ids, list) else None
                    tok_text = tokenizer.decode([tid]) if isinstance(tid, int) else formatted_prompt[pre_offsets[idx][0]:pre_offsets[idx][1]]
                    t = tok_text.strip()
                    is_special = (t.startswith('[') and t.endswith(']')) or (t.startswith('<') and t.endswith('>'))
                    is_ws = re.fullmatch(r"\s+", tok_text) is not None
                    is_punct_only = re.fullmatch(r"\W+", t) is not None if t else True
                    if not is_special and not is_ws and not is_punct_only:
                        chosen = idx
                        break
                decision_index = chosen if chosen is not None else pre_token_indices[-1]
            except Exception:
                decision_index = pre_token_indices[-1]
            if DEBUG_ATTENTION_LOGS:
                try:
                    da, db = pre_offsets[decision_index]
                    dec_txt = formatted_prompt[da:db].replace("\n", "\\n")
                    print(f"[decision_token] using last email token index={decision_index} text='{dec_txt}' span=({da},{db})")
                except Exception:
                    pass
        
        # Extract attention scores and offsets using decoder-only label-conditioned path
        if DEBUG_ATTENTION_LOGS:
            print("[extract_mode] decoder: using label-conditioned attention (predicted label)")
        predicted_label = sentiment_prediction if isinstance(sentiment_prediction, str) else ''
        raw_gen_label = probabilities.get('_raw_output', '').strip()
        if DEBUG_ATTENTION_LOGS:
            try:
                print(f"[forced_label_source] predicted='{predicted_label}' raw_gen='{raw_gen_label}'")
            except Exception:
                pass
        label_text = predicted_label if predicted_label else raw_gen_label
        if label_text:
            offsets, token_scores_dict = extract_forced_label_attention(model, tokenizer, formatted_prompt, label_text, email_start, email_end)
        else:
            offsets, token_scores_dict = extract_generated_label_attention(model, tokenizer, formatted_prompt, email_start, email_end)
        if DEBUG_ATTENTION_LOGS:
            try:
                da, db = offsets[decision_index]
                dec_txt = formatted_prompt[da:db].replace("\n","\\n")
                print(f"[decision_token] index={decision_index} text='{dec_txt}' span=({da},{db})")
            except Exception:
                pass
        
        # Identify the email substring span within the prompt (reusing computed markers)
        # start_marker/end_marker/email_start/email_end already computed above

        # Select token indices overlapping the email substring
        token_indices = [i for i, (a, b) in enumerate(offsets) if not (b <= email_start or a >= email_end)]
        if not token_indices:
            # Fallback to entire text if parsing fails
            token_indices = list(range(len(offsets)))
            email_start = 0
            email_end = len(formatted_prompt)
        if DEBUG_ATTENTION_LOGS:
            print(f"[get_attention] total_tokens={len(offsets)} email_tokens={len(token_indices)} email_span=({email_start},{email_end}) decision_index={decision_index}")

        # Clip token offsets to the email substring coordinates
        clipped_offsets = [(max(a - email_start, 0), max(min(b, email_end) - email_start, 0)) for i, (a, b) in enumerate(offsets) if i in token_indices]

        # Per-token scores (already aligned with model tokens)
        def take(scores):
            return [scores[i] for i in token_indices]
        token_scores_email = {
            'layer_average': take(token_scores_dict['layer_average']),
            'rollout': take(token_scores_dict['rollout']),
        }
        if DEBUG_ATTENTION_LOGS:
            try:
                la = np.array(token_scores_email['layer_average'])
                ro = np.array(token_scores_email['rollout'])
                print(f"[get_attention] email token score stats layer_avg(min={la.min():.4f} max={la.max():.4f}) rollout(min={ro.min():.4f} max={ro.max():.4f})")
                label_tok = token_scores_dict.get('_generated_label_token', '')
                print(f"[attention_source] decoder_label_token='{label_tok}' aggregated=True")
            except Exception:
                pass

        # Build per-sentence aggregation over the email text
        email_text_only = formatted_prompt[email_start:email_end]
        # Simple sentence segmentation: split on ., !, ? while preserving boundaries
        import re
        sentence_spans = []
        for m in re.finditer(r"[^.!?]+[.!?]?\s*", email_text_only):
            a = m.start()
            b = m.end()
            if b > a:
                sentence_spans.append((a, b))
        if not sentence_spans:
            sentence_spans = [(0, len(email_text_only))]

        def aggregate_by_spans(spans, offsets_rel, values):
            agg = [0.0] * len(spans)
            for (ta, tb), val in zip(offsets_rel, values):
                for j, (sa, sb) in enumerate(spans):
                    if not (tb <= sa or ta >= sb):
                        agg[j] += float(val)
            # Normalize
            if agg:
                mn, mx = min(agg), max(agg)
                agg = [(x - mn) / (mx - mn) if mx > mn else 0.5 for x in agg]
            return agg

        sentence_scores = {
            'layer_average': aggregate_by_spans(sentence_spans, clipped_offsets, token_scores_email['layer_average']),
            'rollout': aggregate_by_spans(sentence_spans, clipped_offsets, token_scores_email['rollout']),
        }
        if DEBUG_ATTENTION_LOGS:
            try:
                print(f"[get_attention] sentences={len(sentence_spans)} first_span={sentence_spans[0] if sentence_spans else None}")
            except Exception:
                pass

        # Also compute word-level aggregation for backward compatibility in UI (using proper offsets)
        # Build word spans
        words = text.split()
        word_spans = []
        running = 0
        for w in words:
            idx = text.find(w, running)
            if idx < 0:
                continue
            word_spans.append((idx, idx + len(w)))
            running = idx + len(w)
        # Convert to email_text_only coordinates by mapping the raw text to email_text_only
        # If exact mapping is complex, fallback to sentence-level in UI; keep placeholders consistent length
        word_scores = {
            'layer_average': [0.5] * len(words),
            'rollout': [0.5] * len(words),
        }

        if DEBUG_ATTENTION_LOGS:
            # Preview top tokens by rollout and layer avg
            try:
                def preview_top(scores, name):
                    arr = list(enumerate(scores))
                    arr.sort(key=lambda t: t[1], reverse=True)
                    top = arr[:10]
                    toks = []
                    for idx, val in top:
                        a, b = clipped_offsets[idx]
                        tok = email_text_only[a:b]
                        cleaned = tok.strip()
                        if cleaned == "":
                            continue
                        toks.append((tok.replace("\n", " "), float(val)))
                    print(f"[top_tokens] {name}:", ", ".join([f"{t[0]}({t[1]:.4f})" for t in toks]))
                preview_top(token_scores_email['rollout'], 'rollout')
                preview_top(token_scores_email['layer_average'], 'layer_avg')
            except Exception:
                pass

        return {
            'attention_scores': {
                'layer_average': word_scores['layer_average'],
                'rollout': word_scores['rollout'],
            },
            'token_attention': {
                'offsets': clipped_offsets,
                'layer_average': token_scores_email['layer_average'],
                'rollout': token_scores_email['rollout'],
                'text': text,
            },
            'sentence_attention': {
                'spans': sentence_spans,
                'layer_average': sentence_scores['layer_average'],
                'rollout': sentence_scores['rollout'],
                'text': text,
            },
            'label': sentiment_prediction,
            'probabilities': probabilities,
            'confidence': max(v for k, v in probabilities.items() if not k.startswith('_')),
            'raw_model_output': probabilities.get('_raw_output', ''),
            'used_prompt': probabilities.get('_prompt', ''),
        }
        
    except Exception as e:
        print(f"Error in get_attention: {e}")
        # Fallback: return uniform attention scores
        words = text.split()
        return {
            'attention_scores': {
                'layer_average': [0.5] * len(words),
                'rollout': [0.5] * len(words),
            },
            'sentiment': 'Neutral',
            'probabilities': {'Positive': 0.33, 'Negative': 0.33, 'Neutral': 0.34},
            'confidence': 0.34
        }

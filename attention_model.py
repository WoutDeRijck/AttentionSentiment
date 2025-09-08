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
You are an email classifier specialized in extracting a topic.
You only respond with 1 topic without any chat-based fluff.

I have the following email:

{text}

Categorize the email in one of the following categories, pick the most specific one:
Negative(Negative)
Negative sentiment expressed in this email, customer is not happy. Only use if strong negative sentiment is expressed. This option should only be picked if there is a strong negative sentiment. IGNORE THE SEVERITY AND THE WORDS LIKE ONGOING IMPACT Ignore html tags.
Neutral(Neutral)
Default option, formal or no strong expression of positive/negative feelings. Should be picked most of the time as this is the default.
Positive(Positive)
Positive sentiment expressed in this email, customer expresses happy / good feelings. Should be picked when customer expresses clear positive signals.

Make sure to only respond with 1 topic that is the best fitting.
Say nothing else. For example, do not say: 'Here is the topic.' or "Topic:".
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

            # _pipeline = pipeline(
            #     "text-generation", 
            #     # model="google/gemma-3-1b-it",
            #     model="meta-llama/Llama-3.2-1B",
            #     device=target_device,
            #     torch_dtype=target_dtype,
            #     model_kwargs={"attn_implementation": "eager"}  # Force eager attention for output_attentions
            # )

            _pipeline = pipeline(
                "text-classification",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                device=target_device,
                model_kwargs={"attn_implementation": "eager"}  # Force eager attention for output_attentions
            )
            
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
        a_tilde.append(_row_normalize(A_res))
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
    layer_mats = [att[0].mean(dim=0) for att in attentions]  # list of [seq, seq]
    M = torch.stack(layer_mats, dim=0).mean(dim=0)  # [seq, seq]
    # Decision token row (e.g., -1 for decoder, 0 for encoder [CLS])
    decision_row = M[decision_index]
    scores = decision_row.detach().cpu().numpy()
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
    # Never mask the decision token and ensure we keep useful tokens
    if 0 <= (decision_index if decision_index >= 0 else len(decoded_tokens) + decision_index) < len(decoded_tokens):
        idx = decision_index if decision_index >= 0 else len(decoded_tokens) + decision_index
        noise_mask[idx] = False
    # If masking removes almost everything, disable masking to avoid degenerate 0.5 outputs
    if (np.sum(~noise_mask) < max(2, int(0.1 * len(noise_mask)))):
        noise_mask = None

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
    Predict sentiment using the Gemma model and get the probability distribution.
    
    Args:
        text (str): The input email text to analyze
        
    Returns:
        tuple: (predicted_sentiment, probability_distribution)
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

        if getattr(_pipeline, 'task', None) == 'text-classification':
            # Classifier: logits are [batch, num_labels]; feed raw text
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # [1, num_labels]
            probs = torch.softmax(logits[0], dim=-1)
            # Map id->label and build Positive/Negative (binary). Neutral set to 0.0
            id2label = getattr(model.config, 'id2label', {i: str(i) for i in range(probs.shape[-1])})
            prob_dict = {'Positive': 0.0, 'Negative': 0.0}
            for idx, p in enumerate(probs.tolist()):
                label = id2label.get(idx, '').lower()
                if 'pos' in label:
                    prob_dict['Positive'] += p
                elif 'neg' in label:
                    prob_dict['Negative'] += p
            # Normalize to sum<=1 and add Neutral=0 for UI compatibility
            s = prob_dict['Positive'] + prob_dict['Negative']
            if s > 0:
                prob_dict['Positive'] /= s
                prob_dict['Negative'] /= s
            prob_dict_full = {**prob_dict, 'Neutral': 0.0}
            predicted_sentiment = 'Positive' if prob_dict['Positive'] >= prob_dict['Negative'] else 'Negative'
            return predicted_sentiment, prob_dict_full
        else:
            # Decoder LM: use prompt and last-token logits trick
            formatted_prompt = prompt.format(text=text)
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
            # Use the first token-piece for each label; if missing, back off to substring search
            def first_id(label: str):
                ids = tokenizer.encode(label, add_special_tokens=False)
                return ids[0] if ids else None
            pos_id = first_id("Positive")
            neg_id = first_id("Negative")
            neu_id = first_id("Neutral")
            positive_logit = logits[pos_id].item() if pos_id is not None else -float('inf')
            negative_logit = logits[neg_id].item() if neg_id is not None else -float('inf')
            neutral_logit = logits[neu_id].item() if neu_id is not None else -float('inf')
            sentiment_logits = torch.tensor([positive_logit, negative_logit, neutral_logit], device=logits.device)
            probabilities = torch.softmax(sentiment_logits, dim=0)
            prob_dict = {
                'Positive': probabilities[0].item(),
                'Negative': probabilities[1].item(),
                'Neutral': probabilities[2].item()
            }
            predicted_sentiment = max(prob_dict, key=prob_dict.get)
            # Also provide raw generation to be shown in UI
            try:
                gen = _pipeline(formatted_prompt, max_new_tokens=8, do_sample=False)
                raw_out = gen[0].get('generated_text', '')
                # Trim the prompt prefix so UI shows only the model's continuation
                raw_output = raw_out[len(formatted_prompt):].strip() if raw_out.startswith(formatted_prompt) else raw_out
            except Exception:
                raw_output = ''
            return predicted_sentiment, {**prob_dict, '_raw_output': raw_output, '_prompt': formatted_prompt}
        
    except Exception as e:
        print(f"Error in sentiment prediction with distribution: {e}")
        # Fallback to simple generation
        try:
            response = _pipeline(formatted_prompt, max_new_tokens=10, do_sample=False)
            generated_text = response[0]['generated_text']
            sentiment_prediction = generated_text[len(formatted_prompt):].strip()
            sentiment_prediction = sentiment_prediction.split('\n')[0].strip()
            
            if 'Negative' in sentiment_prediction:
                predicted = 'Negative'
            elif 'Positive' in sentiment_prediction:
                predicted = 'Positive'
            else:
                predicted = 'Neutral'
            
            # Return uniform probabilities as fallback
            prob_dict = {'Positive': 0.33, 'Negative': 0.33, 'Neutral': 0.34}
            prob_dict[predicted] = 0.7  # Give higher confidence to predicted class
            remaining = 0.15
            for key in prob_dict:
                if key != predicted:
                    prob_dict[key] = remaining
                    
            return predicted, {**prob_dict, '_raw_output': generated_text, '_prompt': formatted_prompt}
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return 'Neutral', {'Positive': 0.33, 'Negative': 0.33, 'Neutral': 0.34, '_raw_output': '', '_prompt': ''}


def predict_sentiment(text):
    """
    Predict sentiment using the Gemma model with the defined prompt.
    
    Args:
        text (str): The input email text to analyze
        
    Returns:
        str: Predicted sentiment (Positive, Negative, or Neutral)
    """
    sentiment, _ = predict_sentiment_with_distribution(text)
    return sentiment


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
        
        # Determine decision token index based on model type
        # Encoder-only (e.g., BERT/DistilBERT): use [CLS] at index 0
        # Decoder-only (e.g., GPT/Llama): use last token -1
        try:
            is_encoder_only = getattr(model.config, 'is_encoder_decoder', False) is False and getattr(model.config, 'model_type', '') in ['bert', 'distilbert', 'roberta']
        except Exception:
            is_encoder_only = False
        decision_index = 0 if is_encoder_only else -1

        # Predict sentiment with probabilities using the prompt
        sentiment_prediction, probabilities = predict_sentiment_with_distribution(text)
        
        # For attention extraction, we'll use the formatted prompt to see what the model focuses on
        formatted_prompt = prompt.format(text=text)
        
        # Extract attention scores and offsets from the formatted prompt (per-token)
        offsets, token_scores_dict = extract_attention_scores(model, tokenizer, formatted_prompt, decision_index=decision_index)
        
        # Identify the email substring span within the prompt
        start_marker = "I have the following email:"
        end_marker = "Categorize the email"
        s = formatted_prompt.find(start_marker)
        e = formatted_prompt.find(end_marker, s if s >= 0 else 0)
        email_start = (s + len(start_marker)) if s >= 0 else 0
        email_end = e if e >= 0 else len(formatted_prompt)

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
                        toks.append((email_text_only[a:b], float(val)))
                    print(f"[top_tokens] {name}:", ", ".join([f"{t[0]}({t[1]:.2f})" for t in toks]))
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
                'text': email_text_only,
            },
            'sentence_attention': {
                'spans': sentence_spans,
                'layer_average': sentence_scores['layer_average'],
                'rollout': sentence_scores['rollout'],
                'text': email_text_only,
            },
            'sentiment': sentiment_prediction,
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

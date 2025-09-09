import streamlit as st
from attention_model import get_attention

    # Configure page
st.set_page_config(
    page_title="LLM Attention Classification Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        color: #111;
    }
    
    .input-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #111;
    }
    
    .results-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #111;
    }
    
    .analyze-button {
        display: flex;
        justify-content: center;
        padding: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

def highlight_text_with_attention(text, attention_scores):
    """
    Create HTML with text highlighted based on attention scores.
    Higher attention = darker highlighting.
    """
    words = text.split()
    
    # If no attention scores provided, return original text
    if not attention_scores or len(attention_scores) != len(words):
        return text
    
    # Normalize scores to 0-1 range for consistent coloring
    min_score = min(attention_scores)
    max_score = max(attention_scores)
    if max_score > min_score:
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in attention_scores]
    else:
        normalized_scores = [0.5] * len(attention_scores)
    
    highlighted_words = []
    for word, norm_score, raw_score in zip(words, normalized_scores, attention_scores):
        # Convert score to color intensity (lighter = less attention, darker = more attention)
        intensity = int(255 * (1 - norm_score * 0.8))  # Stronger highlighting
        color = f"rgba(255, {intensity}, {intensity}, 0.9)"
        
        # Clean highlighting without numbers
        highlighted_words.append(
            f'<span style="background-color: {color}; padding: 3px 6px; '
            f'border-radius: 4px; margin: 2px; font-weight: {400 + int(norm_score * 200)};" '
            f'title="Attention Score: {raw_score:.3f}">{word}</span>'
        )
    
    return ' '.join(highlighted_words)

def render_token_attention_html(text, offsets, scores, min_threshold: float = 0.0):
    """
    Create HTML highlighting based on token offsets over the given text.
    Offsets are (start, end) indices relative to the provided text.
    """
    if not offsets or not scores or len(offsets) != len(scores):
        return text
    # Scores assumed to be in [0,1]; if not, normalize
    min_s = min(scores)
    max_s = max(scores)
    if max_s > min_s:
        norm = [(s - min_s) / (max_s - min_s) for s in scores]
    else:
        norm = [0.5] * len(scores)
    html_parts = []
    cursor = 0
    for (start, end), s in zip(offsets, norm):
        # Add plain text before token
        if start > cursor:
            html_parts.append(text[cursor:start])
        span_text = text[start:end]
        if s >= min_threshold:
            intensity = int(255 * (1 - s * 0.8))
            color = f"rgba(255, {intensity}, {intensity}, 0.9)"
            html_parts.append(
                f'<span style="background-color: {color}; color:#111; padding: 2px 3px; '
                f'border-radius: 3px; margin: 0 1px; font-weight: {400 + int(s * 200)};" '
                f'title="Attention: {s:.3f}">{span_text}</span>'
            )
        else:
            html_parts.append(span_text)
        cursor = end
    if cursor < len(text):
        html_parts.append(text[cursor:])
    # Ensure text color inside highlight spans is dark for readability
    out = ''.join(html_parts)
    return out

def render_sentence_attention_html(text, spans, scores):
    """
    Create HTML highlighting sentences based on sentence spans and scores.
    Spans are (start, end) indices relative to the provided text.
    """
    if not spans or not scores or len(spans) != len(scores):
        return text
    min_s = min(scores)
    max_s = max(scores)
    if max_s > min_s:
        norm = [(s - min_s) / (max_s - min_s) for s in scores]
    else:
        norm = [0.5] * len(scores)
    html_parts = []
    cursor = 0
    for (start, end), s in zip(spans, norm):
        if start > cursor:
            html_parts.append(text[cursor:start])
        intensity = int(255 * (1 - s * 0.8))
        color = f"rgba(255, {intensity}, {intensity}, 0.3)"  # lighter for full sentence blocks
        segment = text[start:end]
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f'border-radius: 4px; margin: 0 2px; display: inline;" '
            f'title="Sentence attention: {s:.3f}">{segment}</span>'
        )
        cursor = end
    if cursor < len(text):
        html_parts.append(text[cursor:])
    return ''.join(html_parts)

def compute_word_spans(text: str):
    """
    Return spans (start, end) for contiguous non-whitespace sequences in text.
    Keeps the original text intact; spans map onto the same string.
    """
    import re
    return [m.span() for m in re.finditer(r"\S+", text)]

def aggregate_token_scores_to_spans(spans, token_offsets, token_scores):
    """
    Aggregate token scores onto arbitrary spans using overlap-sum, then min-max normalize.
    """
    if not spans or not token_offsets or not token_scores:
        return []
    agg = [0.0] * len(spans)
    for (ta, tb), s in zip(token_offsets, token_scores):
        for i, (sa, sb) in enumerate(spans):
            if not (tb <= sa or ta >= sb):
                agg[i] += float(s)
    if agg:
        mn, mx = min(agg), max(agg)
        if mx > mn:
            agg = [(x - mn) / (mx - mn) for x in agg]
        else:
            agg = [0.5 for _ in agg]
    return agg

def max_token_scores_to_spans(spans, token_offsets, raw_scores, norm_scores):
    """
    For each span, take the maximum token score among overlapping tokens.
    Returns (norm_list, raw_list) aligned with spans.
    """
    norm_out = [0.0] * len(spans)
    raw_out = [0.0] * len(spans)
    if not spans or not token_offsets or not raw_scores:
        return norm_out, raw_out
    for i, (sa, sb) in enumerate(spans):
        max_norm = 0.0
        max_raw = 0.0
        for (ta, tb), rs, ns in zip(token_offsets, raw_scores, norm_scores if norm_scores else raw_scores):
            if not (tb <= sa or ta >= sb):
                if ns > max_norm:
                    max_norm = float(ns)
                    max_raw = float(rs)
        norm_out[i] = max_norm
        raw_out[i] = max_raw
    return norm_out, raw_out

def render_span_attention_html(text, spans, scores, min_threshold: float = 0.0, raw_scores=None):
    """
    Highlight arbitrary spans with their scores. Overlap is not expected here;
    spans should be non-overlapping and ordered.
    """
    if not spans or not scores or len(spans) != len(scores):
        return text
    html_parts = []
    cursor = 0
    # normalize already done by caller for scores; still guard for 0..1
    s_min, s_max = min(scores), max(scores)
    if s_max > s_min:
        norm = [(s - s_min) / (s_max - s_min) for s in scores]
    else:
        norm = [0.5] * len(scores)
    for (start, end), s in zip(spans, norm):
        if start > cursor:
            html_parts.append(text[cursor:start])
        segment = text[start:end]
        if s >= min_threshold:
            intensity = int(255 * (1 - s * 0.8))
            color = f"rgba(255, {intensity}, {intensity}, 0.9)"
            title_val = s
            if raw_scores and len(raw_scores) == len(spans):
                try:
                    title_val = float(raw_scores[spans.index((start, end))])
                except Exception:
                    title_val = s
            html_parts.append(
                f'<span style="background-color: {color}; color:#111; padding: 2px 3px; '
                f'border-radius: 3px; margin: 0 1px; font-weight: {400 + int(s * 200)};" '
                f'title="Attention: {title_val:.3f}">{segment}</span>'
            )
        else:
            html_parts.append(segment)
        cursor = end
    if cursor < len(text):
        html_parts.append(text[cursor:])
    return ''.join(html_parts)

def _build_inside_tag_mask(text: str):
    """Return a boolean list per character: True if inside <...> HTML-like tag."""
    inside = False
    mask = []
    for ch in text:
        if ch == '<':
            inside = True
        mask.append(inside)
        if ch == '>':
            inside = False
    return mask

def _alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    total = sum(1 for c in s if not c.isspace())
    if total == 0:
        return 0.0
    alpha = sum(1 for c in s if c.isalpha())
    return alpha / total

def _is_structural_noise(segment: str) -> bool:
    """Heuristic structural filter for UI-only suppression: tags/css/low-alpha and boilerplate labels."""
    if not segment:
        return True
    s = segment.strip().lower()
    if s in {"subject:", "description:", "subject", "description"}:
        return True
    if 'color' in s or 'rgb(' in s:
        return True
    return _alpha_ratio(s) < 0.4

def main():
    # Main header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🔍 LLM Attention Classification Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

    # Persist last analysis to allow method switching without recomputation
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None

    # Sidebar for attention visualization controls
    with st.sidebar:
        st.header("Visualization Controls")
        
        method = st.radio(
            "Attention Method",
            options=["Layer Average", "Rollout"],
            horizontal=False,
            index=1,
            key="attention_method"
        )
        

        aggregate_words = st.checkbox(
            "Aggregate token attentions to words",
            value=True,
            help="When enabled, token scores are merged per word for display."
        )
        
        # Percentage sliders for highlighting thresholds
        if aggregate_words:
            top_pct = st.slider("Top words (%)", 1, 100, 30, 1)
        else:
            top_pct = st.slider("Top tokens (%)", 1, 100, 20, 1)

    # Create three columns for layout
    left_col, center_col, right_col = st.columns([2, 1, 2])

    with left_col:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("📧 Input Email")
        
        # Text area for email input
        email_text = st.text_area(
            "Enter email text:",
            height=350,
            placeholder="Paste your email text here..."
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with center_col:
        st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
        
        # Add some vertical spacing
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        
        analyze_button = st.button(
            "🔍 Get Attention",
            type="primary",
            use_container_width=True,
            help="Click to analyze attention"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.subheader("🎯 Attention Visualization")
        
        # Decide what to render: new result (on click) or last cached result
        result_data = None
        last_text = st.session_state.analysis['text'] if st.session_state.analysis else None
        if analyze_button and email_text.strip():
            with st.spinner("Computing attention..."):
                attention_result = get_attention(email_text)
                st.session_state.analysis = {'text': email_text, 'result': attention_result}
                result_data = attention_result
        elif st.session_state.analysis and st.session_state.analysis.get('result'):
            result_data = st.session_state.analysis['result']
            if email_text and last_text and email_text != last_text:
                st.info("Showing results for the previous input. Click 'Get Attention' to recompute for current text.")

        if result_data:
            # Display token/sentence attention with correct mapping FIRST
            token_att = result_data.get('token_attention')
            if token_att:
                method_key = {"Layer Average": "layer_average", "Rollout": "rollout"}[method]

                text_to_show = token_att.get('text', '')
                offsets = token_att.get('offsets', [])
                scores = token_att.get(method_key, [])

                if not aggregate_words:
                    # Token-level flow
                    cutoff = 1.0
                    norm_scores = []
                    if scores:
                        # Apply structural filtering to visualization: zero-out non-informative tokens
                        inside_mask_chars = _build_inside_tag_mask(text_to_show)
                        vis_scores = []
                        for (start, end), s in zip(offsets, scores):
                            seg = text_to_show[start:end]
                            cleaned = seg.strip()
                            inside_tag = any(inside_mask_chars[i] for i in range(start, min(end, len(inside_mask_chars))))
                            vis_scores.append(0.0 if (inside_tag or _is_structural_noise(cleaned)) else float(s))
                        s_min, s_max = min(vis_scores), max(vis_scores)
                        norm_scores = [(s - s_min) / (s_max - s_min) if s_max > s_min else 0.5 for s in vis_scores]
                        k = max(1, int(round(top_pct / 100.0 * len(norm_scores))))
                        sorted_vals = sorted(norm_scores, reverse=True)
                        cutoff = sorted_vals[k - 1] if k - 1 < len(sorted_vals) else sorted_vals[-1]
                    highlighted_html = render_token_attention_html(text_to_show, offsets, vis_scores if scores else scores, min_threshold=cutoff)

                    st.markdown(
                        f'<div style="line-height: 1.8; font-size: 16px; padding: 15px; '
                        f'background-color: #f8f9fa; color:#111; border-radius: 8px; margin: 10px 0; white-space: pre-wrap;">{highlighted_html}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    # Word-aggregated flow
                    word_spans = compute_word_spans(text_to_show)
                    # Precompute token norm for consistent thresholding
                    norm_scores = []
                    if scores:
                        s_min, s_max = min(scores), max(scores)
                        norm_scores = [(s - s_min) / (s_max - s_min) if s_max > s_min else 0.5 for s in scores]
                    # Structural filtering at token level BEFORE aggregation
                    inside_mask_chars = _build_inside_tag_mask(text_to_show)
                    filtered_token_scores = []
                    for (start, end), s in zip(offsets, scores):
                        seg = text_to_show[start:end]
                        cleaned = seg.strip()
                        inside_tag = any(inside_mask_chars[i] for i in range(start, min(end, len(inside_mask_chars))))
                        filtered_token_scores.append(0.0 if (inside_tag or _is_structural_noise(cleaned)) else float(s))
                    # Aggregate by sum using filtered tokens
                    word_scores = aggregate_token_scores_to_spans(word_spans, offsets, filtered_token_scores)
                    # Percentile cutoff on words
                    cutoff = 1.0
                    if word_scores:
                        ws_sorted = sorted(word_scores, reverse=True)
                        k = max(1, int(round(top_pct / 100.0 * len(ws_sorted))))
                        cutoff = ws_sorted[k - 1] if k - 1 < len(ws_sorted) else ws_sorted[-1]
                    # Structural filtering on display as well
                    inside_mask = _build_inside_tag_mask(text_to_show)
                    filtered_spans = []
                    filtered_scores = []
                    for (a, b), s in zip(word_spans, word_scores):
                        seg = text_to_show[a:b]
                        inside_tag = any(inside_mask[i] for i in range(a, min(b, len(inside_mask))))
                        if inside_tag or _is_structural_noise(seg):
                            continue
                        filtered_spans.append((a, b))
                        filtered_scores.append(s)
                    highlighted_html = render_span_attention_html(
                        text_to_show,
                        filtered_spans if filtered_spans else word_spans,
                        filtered_scores if filtered_scores else word_scores,
                        min_threshold=cutoff
                    )
                    st.markdown(
                        f'<div style="line-height: 1.8; font-size: 16px; padding: 15px; '
                        f'background-color: #f8f9fa; color:#111; border-radius: 8px; margin: 10px 0; white-space: pre-wrap;">{highlighted_html}</div>',
                        unsafe_allow_html=True
                    )

            # Display prediction distribution AFTER attention visualization
            label = result_data.get('label', 'Unknown')
            probabilities = result_data.get('probabilities', {})
            if probabilities:
                st.write("Model's confidence for each label:")
                col1, col2, col3 = st.columns(3)
                colors = {'Attention_needed': "#ff7f0e", 'Default': "#1f77b4", 'Escalate': "#d62728"}
                for i, (lbl, prob) in enumerate([
                    ('Attention_needed', probabilities.get('Attention_needed', 0)),
                    ('Default', probabilities.get('Default', 0)),
                    ('Escalate', probabilities.get('Escalate', 0))
                ]):
                    col = [col1, col2, col3][i]
                    with col:
                        is_predicted = lbl == label
                        border_style = f"border: 2px solid {colors[lbl]};" if is_predicted else "border: 1px solid #ddd;"
                        st.markdown(f'''
                            <div style="text-align: center; padding: 10px; {border_style} 
                                        border-radius: 8px; background-color: {colors[lbl]}10;">
                                <h4 style="margin: 0; color: {colors[lbl]};">
                                    {lbl}
                                </h4>
                                <div style="font-size: 24px; font-weight: bold; color: {colors[lbl]};">
                                    {prob:.1%}
                                </div>
                                <div style="background-color: #f0f0f0; border-radius: 10px; height: 8px; margin: 8px 0;">
                                    <div style="background-color: {colors[lbl]}; height: 8px; 
                                                border-radius: 10px; width: {prob*100}%;"></div>
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)
                
        elif analyze_button and not email_text.strip():
            st.error("Please enter some email text to analyze.")
        elif not result_data:
            st.info("👆 Enter email text and click 'Get Attention' to see results")
            st.write("*Highlighted text will appear here*")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

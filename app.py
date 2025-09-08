import streamlit as st
from attention_model import get_attention

# Configure page
st.set_page_config(
    page_title="LLM Attention Sentiment Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
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

def main():
    # Main header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🔍 LLM Attention Sentiment Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

    # Persist last analysis to allow method switching without recomputation
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None

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
            # Display prediction distribution
            sentiment = result_data.get('sentiment', 'Unknown')
            probabilities = result_data.get('probabilities', {})
            raw_output = result_data.get('raw_model_output', '')
            used_prompt = result_data.get('used_prompt', '')
            if probabilities:
                st.subheader("📊 Prediction Distribution")
                st.write("Model's confidence for each sentiment category:")
                col1, col2, col3 = st.columns(3)
                colors = {'Positive': "#28a745", 'Negative': "#dc3545", 'Neutral': "#ffc107"}
                for i, (sentiment_type, prob) in enumerate([
                    ('Positive', probabilities.get('Positive', 0)),
                    ('Negative', probabilities.get('Negative', 0)),
                    ('Neutral', probabilities.get('Neutral', 0))
                ]):
                    col = [col1, col2, col3][i]
                    with col:
                        is_predicted = sentiment_type == sentiment
                        border_style = f"border: 2px solid {colors[sentiment_type]};" if is_predicted else "border: 1px solid #ddd;"
                        st.markdown(f'''
                            <div style="text-align: center; padding: 10px; {border_style} 
                                        border-radius: 8px; background-color: {colors[sentiment_type]}10;">
                                <h4 style="margin: 0; color: {colors[sentiment_type]};">
                                    {sentiment_type}
                                </h4>
                                <div style="font-size: 24px; font-weight: bold; color: {colors[sentiment_type]};">
                                    {prob:.1%}
                                </div>
                                <div style="background-color: #f0f0f0; border-radius: 10px; height: 8px; margin: 8px 0;">
                                    <div style="background-color: {colors[sentiment_type]}; height: 8px; 
                                                border-radius: 10px; width: {prob*100}%;"></div>
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)

            # Show raw model output and the prompt used
            if raw_output:
                st.subheader("🧪 Raw model output")
                st.code(raw_output, language="text")
            if used_prompt:
                with st.expander("Show used prompt"):
                    st.code(used_prompt, language="text")

            # Display token/sentence attention with correct mapping
            token_att = result_data.get('token_attention')
            if token_att:
                st.subheader("🎯 Attention Visualization")
                st.write("Token-level highlighting uses exact tokenizer offsets. Adjust the threshold to show only the most attended tokens.")
                method = st.radio(
                    "Method",
                    options=["Layer Average", "Rollout"],
                    horizontal=True,
                    index=1,
                    key="attention_method"
                )
                method_key = {"Layer Average": "layer_average", "Rollout": "rollout"}[method]

                hide_noise = st.checkbox(
                    "Hide punctuation and common stopwords",
                    value=True,
                    help="When enabled, punctuation and very common words are hidden from the top-tokens list."
                )

                text_to_show = token_att.get('text', '')
                offsets = token_att.get('offsets', [])
                scores = token_att.get(method_key, [])

                # Normalize scores and compute percentile-based cutoff
                cutoff = 1.0
                norm_scores = []
                if scores:
                    s_min, s_max = min(scores), max(scores)
                    norm_scores = [(s - s_min) / (s_max - s_min) if s_max > s_min else 0.5 for s in scores]
                    top_pct = st.slider(
                        "Top tokens (%)",
                        min_value=1,
                        max_value=100,
                        value=20,
                        step=1,
                        help="Highlight the top X% tokens by normalized attention"
                    )
                    k = max(1, int(round(top_pct / 100.0 * len(norm_scores))))
                    sorted_vals = sorted(norm_scores, reverse=True)
                    cutoff = sorted_vals[k - 1] if k - 1 < len(sorted_vals) else sorted_vals[-1]
                highlighted_html = render_token_attention_html(text_to_show, offsets, scores, min_threshold=cutoff)

                st.markdown(
                    f'<div style="line-height: 1.8; font-size: 16px; padding: 15px; '
                    f'background-color: #f8f9fa; color:#111; border-radius: 8px; margin: 10px 0; white-space: pre-wrap;">{highlighted_html}</div>',
                    unsafe_allow_html=True
                )

                # Show top tokens above threshold
                st.subheader("🔝 Highest-attended tokens")
                # Use the same percentile-based cutoff for listing top tokens
                if scores:
                    top = []
                    # Basic English stopword list (lightweight, no extra deps)
                    stopwords = {
                        'the','is','am','are','was','were','be','been','being','a','an','and','or','but','if','then','so','to','of','in','on','for','with','by','from','as','at','this','that','these','those','it','its','your','you','yours','we','our','ours','they','their','theirs','he','him','his','she','her','hers','i','me','my','mine','do','does','did','have','has','had','will','would','can','could','should','shall','may','might','there','here','hi','hello','thanks'
                    }
                    import re
                    for (start, end), s_norm, s_raw in zip(offsets, norm_scores, scores):
                        if s_norm >= cutoff:
                            tok_text = text_to_show[start:end]
                            cleaned = tok_text.strip()
                            is_punct_only = len(cleaned) > 0 and re.fullmatch(r"\W+", cleaned) is not None
                            is_stop = cleaned.lower() in stopwords
                            if hide_noise and (is_punct_only or is_stop):
                                continue
                            top.append((tok_text, s_norm, s_raw))
                    top.sort(key=lambda t: t[1], reverse=True)
                    if top:
                        chips = " ".join(
                            [
                                f'<span style="background:#eef;color:#111;border:1px solid #ccd;padding:2px 6px;border-radius:12px;margin:2px;display:inline-block;">{tok} <span style="opacity:0.7">({s:.2f})</span></span>'
                                for tok, s, _ in top[:25]
                            ]
                        )
                        st.markdown(f'<div style="padding: 4px 0;">{chips}</div>', unsafe_allow_html=True)
                    else:
                        st.write("No tokens above current threshold.")
                
        elif analyze_button and not email_text.strip():
            st.error("Please enter some email text to analyze.")
        elif not result_data:
            st.info("👆 Enter email text and click 'Get Attention' to see results")
            st.write("*Highlighted text will appear here*")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

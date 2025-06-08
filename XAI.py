import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="HateBERT XAI Explorer", page_icon="🔍", layout="wide")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from captum.attr import IntegratedGradients, Saliency
import shap
import numpy as np
import streamlit.components.v1 as components
import matplotlib
from lime.lime_text import LimeTextExplainer
from bertviz import head_view
import html
import json
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# === Caching for Performance ===
@st.cache_resource
def load_model():
    """Load model and tokenizer with caching"""
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
    model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT")
    model.eval()
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
    return tokenizer, model, pipe

# === Load Model ===
tokenizer, model, pipe = load_model()

# === Token Attribution Visualization ===
def visualize_token_attributions(tokens, scores, title="Attributions"):
    """Visualize token attributions with color coding"""
    if not scores or len(tokens) != len(scores):
        st.warning(f"Token-score mismatch for {title}: {len(tokens)} tokens vs {len(scores)} scores")
        return
    
    scores = np.array(scores)
    
    # Handle edge case where all scores are the same
    if scores.max() == scores.min():
        scores = np.zeros_like(scores)
    
    # Normalize scores
    score_range = scores.max() - scores.min()
    if score_range > 0:
        norm_scores = (scores - scores.min()) / score_range
    else:
        norm_scores = np.zeros_like(scores)
    
    cmap = matplotlib.cm.get_cmap('RdYlBu_r')  # Red for negative, Blue for positive

    def colorize(token, score, norm_score):
        rgba = cmap(float(norm_score))
        color = matplotlib.colors.rgb2hex(rgba)
        # Clean up special tokens
        display_token = token.replace('##', '').replace('[CLS]', '⟨CLS⟩').replace('[SEP]', '⟨SEP⟩')
        return f"<span style='background-color:{color}; padding:2px; margin:1px; border-radius:4px; font-weight:bold' title='Score: {score:.4f}'>{html.escape(display_token)}</span>"

    html_content = " ".join([colorize(t, s, ns) for t, s, ns in zip(tokens, scores, norm_scores)])
    
    st.markdown(f"#### {title}")
    st.markdown(f"*Score range: {scores.min():.4f} to {scores.max():.4f}*")
    components.html(f"""
        <div style='font-family:monospace; font-size:16px; line-height:1.8; padding:10px; border:1px solid #ddd; border-radius:8px; background-color:#f9f9f9;'>
            {html_content}
        </div>
    """, height=200)

# === Prediction Display ===
def display_predictions(text):
    """Display model predictions"""
    predictions = pipe(text)
    if isinstance(predictions, list) and isinstance(predictions[0], list):
        predictions = predictions[0]
    
    st.subheader("🎯 Model Predictions")
    for pred in predictions:
        confidence = pred['score']
        label = pred['label']
        bar_color = "#ff6b6b" if label == "HATE" else "#51cf66"
        st.metric(label=f"{label}", value=f"{confidence:.3f}")

# === App ===
st.title("🔍 HateBERT XAI Explorer")
st.markdown("Explore different explainability techniques for hate speech detection using HateBERT.")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
methods_to_run = st.sidebar.multiselect(
    "Select XAI Methods to Run:",
    ["Model Predictions", "Integrated Gradients", "Saliency Map", "SHAP", "LIME", "Attention Visualization", "Export Embeddings"],
    default=["Model Predictions", "Integrated Gradients", "SHAP"]
)

# Text input
text = st.text_area(
    "Enter a sentence for analysis:", 
    "I hate you and everyone like you.",
    help="Enter text to analyze for hate speech and see explanations from different XAI methods."
)

if text and len(text.strip()) > 0:
    # Progress bar
    progress_bar = st.progress(0)
    total_methods = len(methods_to_run)
    current_step = 0
    
    # Tokenize once
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    try:
        # Model Predictions
        if "Model Predictions" in methods_to_run:
            display_predictions(text)
            current_step += 1
            progress_bar.progress(current_step / total_methods)
        
        # Prepare inputs for attribution methods
        inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
        target_class = 0  # Assuming 0 is hate class
        
        def forward_func(inputs_embeds):
            attention_mask = torch.ones(inputs_embeds.size()[:-1]).to(inputs_embeds.device)
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probs[:, target_class]

        # === Integrated Gradients ===
        if "Integrated Gradients" in methods_to_run:
            st.subheader("🎯 Integrated Gradients")
            with st.spinner("Computing Integrated Gradients..."):
                ig = IntegratedGradients(forward_func)
                attributions_ig, delta = ig.attribute(inputs_embeds, n_steps=50, return_convergence_delta=True)
                scores_ig = attributions_ig.sum(dim=-1).squeeze().detach().cpu().numpy().tolist()
                visualize_token_attributions(tokens, scores_ig, "Integrated Gradients")
                st.info(f"Convergence delta: {delta.item():.6f} (lower is better)")
            current_step += 1
            progress_bar.progress(current_step / total_methods)

        # === Saliency Map ===
        if "Saliency Map" in methods_to_run:
            st.subheader("🌡️ Saliency Map")
            with st.spinner("Computing Saliency Map..."):
                saliency = Saliency(forward_func)
                grads = saliency.attribute(inputs_embeds)
                grads_scores = grads.sum(dim=-1).squeeze().detach().cpu().numpy().tolist()
                visualize_token_attributions(tokens, grads_scores, "Saliency Map")
            current_step += 1
            progress_bar.progress(current_step / total_methods)

        # === SHAP ===
        if "SHAP" in methods_to_run:
            st.subheader("🧩 SHAP Attribution")
            with st.spinner("Computing SHAP values..."):
                try:
                    explainer = shap.Explainer(pipe)
                    shap_values = explainer([text])
                    shap_tokens = shap_values.data[0]
                    
                    shap_scores_raw = shap_values.values[0]
                    if isinstance(shap_scores_raw, np.ndarray) and shap_scores_raw.ndim > 1:
                        shap_scores = shap_scores_raw.sum(axis=1).tolist()
                    else:
                        shap_scores = shap_scores_raw.tolist()
                    
                    visualize_token_attributions(shap_tokens, shap_scores, "SHAP Attribution")
                except Exception as e:
                    st.error(f"SHAP computation failed: {str(e)}")
            current_step += 1
            progress_bar.progress(current_step / total_methods)

        # === LIME ===
        if "LIME" in methods_to_run:
            st.subheader("🍋 LIME Attribution")
            with st.spinner("Computing LIME explanations..."):
                try:
                    # Get model labels
                    dummy_output = pipe("test")
                    if isinstance(dummy_output, list) and isinstance(dummy_output[0], list):
                        dummy_output = dummy_output[0]
                    
                    labels_in_model = sorted([output['label'] for output in dummy_output])
                    label_to_index = {label: idx for idx, label in enumerate(labels_in_model)}

                    def classifier_fn(texts):
                        predictions = []
                        for t in texts:
                            outputs = pipe(t)
                            if isinstance(outputs, list) and isinstance(outputs[0], list):
                                outputs = outputs[0]
                            scores = [0.0] * len(labels_in_model)
                            for output in outputs:
                                idx = label_to_index[output['label']]
                                scores[idx] = output['score']
                            predictions.append(scores)
                        return np.array(predictions)

                    lime_explainer = LimeTextExplainer(class_names=labels_in_model)
                    lime_exp = lime_explainer.explain_instance(
                        text_instance=text,
                        classifier_fn=classifier_fn,
                        num_features=min(len(tokens), 20),  # Limit features for performance
                        labels=[0]
                    )
                    
                    lime_weights = dict(lime_exp.as_list(label=0))
                    # Map LIME weights to tokens (approximate)
                    words = text.split()
                    lime_scores = []
                    for token in tokens:
                        token_clean = token.replace('##', '')
                        score = 0.0
                        for word, weight in lime_weights.items():
                            if word.lower() in token_clean.lower() or token_clean.lower() in word.lower():
                                score = weight
                                break
                        lime_scores.append(score)
                    
                    visualize_token_attributions(tokens, lime_scores, "LIME Attribution")
                except Exception as e:
                    st.error(f"LIME computation failed: {str(e)}")
            current_step += 1
            progress_bar.progress(current_step / total_methods)

        # === BertViz Attention ===
        if "Attention Visualization" in methods_to_run:
            st.subheader("🧠 Attention Head Visualization")
            with st.spinner("Loading attention visualization..."):
                try:
                    from transformers import BertModel
                    
                    # Load base BERT model with attention outputs
                    base_model = BertModel.from_pretrained("GroNLP/hateBERT", output_attentions=True)
                    inputs_for_attn = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)  # Shorter for visualization
                    
                    with torch.no_grad():
                        attn_outputs = base_model(**inputs_for_attn)
                    
                    attention = attn_outputs.attentions
                    tokens_for_viz = tokenizer.convert_ids_to_tokens(inputs_for_attn["input_ids"][0])
                    
                    # Check if we have valid attention and tokens
                    if attention and len(tokens_for_viz) > 0:
                        # Generate head view
                        html_head_view = head_view(attention, tokens_for_viz, html_action='return')
                        
                        # Check if head_view returned valid HTML
                        if html_head_view and hasattr(html_head_view, 'data'):
                            components.html(html_head_view.data, height=800, scrolling=True)
                        else:
                            # Fallback: Create simple attention heatmap
                            st.warning("BertViz head_view failed, showing simple attention summary instead.")
                            
                            # Show attention statistics
                            avg_attention = torch.stack(attention).mean(dim=0)  # Average across layers
                            avg_attention = avg_attention.mean(dim=1)  # Average across heads
                            avg_attention = avg_attention.squeeze().detach().cpu().numpy()
                            
                            # Create simple visualization
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(12, 8))
                            im = ax.imshow(avg_attention, cmap='Blues', aspect='auto')
                            ax.set_xticks(range(len(tokens_for_viz)))
                            ax.set_yticks(range(len(tokens_for_viz)))
                            ax.set_xticklabels(tokens_for_viz, rotation=45, ha='right')
                            ax.set_yticklabels(tokens_for_viz)
                            ax.set_title('Average Attention Weights')
                            plt.colorbar(im)
                            st.pyplot(fig)
                    else:
                        st.error("Could not extract attention weights from the model.")
                        
                except ImportError:
                    st.error("BertViz not installed. Install with: pip install bertviz")
                except Exception as e:
                    st.error(f"Attention visualization failed: {str(e)}")
                    st.info("This might be due to model compatibility issues with BertViz. Try with a shorter text.")
            current_step += 1
            progress_bar.progress(current_step / total_methods)

        # === Export Embeddings ===
        if "Export Embeddings" in methods_to_run:
            st.subheader("📤 Export Embeddings & Data")
            
            # Get embeddings
            embeddings = inputs_embeds.squeeze().detach().cpu().numpy()
            
            # Create comprehensive export data
            export_data = {
                "text": text,
                "tokens": tokens,
                "embeddings": embeddings.tolist(),
                "model": "GroNLP/hateBERT",
                "predictions": pipe(text)[0] if isinstance(pipe(text), list) else pipe(text),
                "metadata": {
                    "model_max_length": tokenizer.model_max_length,
                    "vocab_size": tokenizer.vocab_size,
                    "embedding_dim": embeddings.shape[-1],
                    "sequence_length": len(tokens)
                }
            }
            
            # Add attribution scores if they were computed
            if "Integrated Gradients" in methods_to_run and 'scores_ig' in locals():
                export_data["attributions"] = {
                    "integrated_gradients": scores_ig
                }
            
            export_json = json.dumps(export_data, indent=2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download Complete Analysis JSON",
                    data=export_json,
                    file_name=f"hatebert_analysis_{hash(text) % 10000}.json",
                    mime="application/json"
                )
            
            with col2:
                # Also provide CSV export for easier analysis
                import pandas as pd
                df_data = {
                    "token": tokens,
                    "token_id": inputs["input_ids"][0].tolist(),
                }
                
                # Add attribution scores if available
                if "Integrated Gradients" in methods_to_run and 'scores_ig' in locals():
                    df_data["ig_attribution"] = scores_ig
                if "Saliency Map" in methods_to_run and 'grads_scores' in locals():
                    df_data["saliency_score"] = grads_scores
                
                df = pd.DataFrame(df_data)
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download Token Analysis CSV",
                    data=csv_data,
                    file_name=f"hatebert_tokens_{hash(text) % 10000}.csv",
                    mime="text/csv"
                )
            
            st.info("💡 **Analysis Options:**\n"
                   "- Use the JSON file for custom analysis with the embedding vectors\n"
                   "- Use the CSV file for quick token-level analysis in Excel/Python\n"
                   "- Load embeddings into tools like TensorBoard, UMAP, or t-SNE for visualization")
            
            # Show embedding statistics
            with st.expander("📈 Embedding Statistics"):
                st.write(f"**Embedding Shape:** {embeddings.shape}")
                st.write(f"**Mean Embedding Value:** {embeddings.mean():.6f}")
                st.write(f"**Std Embedding Value:** {embeddings.std():.6f}")
                st.write(f"**Min/Max Values:** {embeddings.min():.6f} / {embeddings.max():.6f}")
            
            current_step += 1
            progress_bar.progress(current_step / total_methods)
        
        progress_bar.progress(1.0)
        st.success("Analysis complete!")
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.info("Try using shorter text or check your internet connection for model loading.")

else:
    st.info("👆 Enter some text above to start the analysis!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit • HateBERT model from Hugging Face • Various XAI libraries")
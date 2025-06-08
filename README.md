# XAI-on-HateBERT
# 🔍 HateBERT XAI Explorer

A Streamlit-based explainable AI (XAI) interface for exploring predictions from [HateBERT](https://huggingface.co/GroNLP/hateBERT) — a BERT-based model fine-tuned to detect hate speech. This app enables visual explanation of model predictions using attention visualization, gradient-based attribution, and model-agnostic interpretation techniques.

---

## 💡 Features

- **Attention Head Visualization** (BertViz integration)
- **Integrated Gradients** and **Saliency Maps** (via Captum)
- **SHAP Explanations** (Token-level influence visualization)
- **LIME Explanations** (Model-agnostic local explanation)
- **Embedding Export** for [TensorBoard Projector](https://projector.tensorflow.org/)

---

## 📦 Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- Streamlit
- Captum
- SHAP
- LIME
- matplotlib
- scikit-learn (for dimensionality reduction if used)

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/hatebert-xai-explorer.git
cd hatebert-xai-explorer

# Create and activate virtual environment (Windows example)
python -m venv hbenv
hbenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run XAI_Pretrained.py --server.runOnSave=false  

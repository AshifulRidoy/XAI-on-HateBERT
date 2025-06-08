from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer

# Load the fine-tuned hateBERT model
model_name = "fahad1247/hateBERT-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize the explainer
explainer = SequenceClassificationExplainer(
    model=model,
    tokenizer=tokenizer
)

# Input text to explain
text = "I hate you so much!"

# Get word attributions
word_attributions = explainer(text)

# Print word attributions
for word, score in word_attributions:
    print(f"{word:>12} : {score:.4f}")

# Print predicted label
pred_class = explainer.predicted_class_name
print(f"\nPredicted Class: {pred_class}")
explainer.visualize()


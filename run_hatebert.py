from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the fine-tuned hateBERT model
model_name = "model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set up the pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Test the classifier
while True:
    text = input("Enter text (or type 'exit' to quit): ")
    if text.lower() == 'exit':
        break
    result = classifier(text)
    print(result)

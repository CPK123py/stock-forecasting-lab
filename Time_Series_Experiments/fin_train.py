import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset


news_data = pd.read_csv("som_news.csv")
df = pd.DataFrame(news_data)

# For a sentiment analysis task, you'd need labels
# Since your data doesn't include sentiment labels, let's add dummy labels for demonstration
# In a real scenario, you'd need to label your data or use unsupervised learning
# 0 = negative, 1 = neutral, 2 = positive
df['label'] = 1  # Assuming all are neutral for demonstration

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 3: Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Step 4: Load pre-trained FinBERT model and tokenizer
# Using the FinBERT model pre-trained for financial sentiment analysis
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["news"], padding="max_length", truncation=True, max_length=128)

# Step 6: Tokenize the datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Step 7: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Step 8: Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

# Step 9: Train the model
trainer.train()

# Step 10: Save the fine-tuned model
model.save_pretrained("./finbert-zomato-news")
tokenizer.save_pretrained("./finbert-zomato-news")

# Step 11: Example of loading and using the saved model
def predict_sentiment(news_text):
    inputs = tokenizer(news_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_label = torch.argmax(predictions, dim=-1).item()
    
    return {
        "text": news_text,
        "sentiment": sentiment_labels[predicted_label],
        "confidence": predictions[0][predicted_label].item()
    }

# Example prediction
sample_news = "Zomato - Scrutinizer's Report of Postal Ballot and Voting Results"
sentiment_prediction = predict_sentiment(sample_news)
print(f"Text: {sentiment_prediction['text']}")
print(f"Sentiment: {sentiment_prediction['sentiment']}")
print(f"Confidence: {sentiment_prediction['confidence']:.4f}")
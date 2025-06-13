from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import nltk
from evaluate import load

# Download NLTK tokenizer
import nltk
nltk.download('punkt')

# Load dataset
raw_datasets = load_dataset("xsum")

# Load tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Preprocess function
def preprocess_function(examples):
    inputs = tokenizer(examples['document'], max_length=1024, truncation=True, padding="max_length")
    targets = tokenizer(examples['summary'], max_length=128, truncation=True, padding="max_length")
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': targets['input_ids']
    }

# Apply preprocessing
processed_datasets = raw_datasets.map(preprocess_function, batched=True)

# Load model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,  # This argument is available in Seq2SeqTrainingArguments
    fp16=True,  # Use mixed precision training
    logging_dir='./logs',
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Load ROUGE metric
rouge_metric = load("rouge")

# Define compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    return result

# Split dataset into train and validation sets
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate model on the test set
test_results = trainer.evaluate(eval_dataset=processed_datasets["test"])

# Print evaluation results
print(test_results)

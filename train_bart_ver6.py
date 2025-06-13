import logging
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, GenerationConfig
import nltk
from evaluate import load

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK tokenizer
nltk.download('punkt')

# Load dataset
raw_datasets = load_dataset("xsum")

# Sample a subset of the dataset before preprocessing
train_subset_raw = raw_datasets["train"].shuffle(seed=42).select(range(5000))  # Select 5000 samples for training
eval_subset_raw = raw_datasets["validation"].shuffle(seed=42).select(range(1000))  # Select 1000 samples for evaluation

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

# Apply preprocessing to the subsets
train_subset = train_subset_raw.map(preprocess_function, batched=True, remove_columns=train_subset_raw.column_names)
eval_subset = eval_subset_raw.map(preprocess_function, batched=True, remove_columns=eval_subset_raw.column_names)

# Load model and set generation config
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.bos_token_id = tokenizer.bos_token_id

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
    predict_with_generate=True,
    fp16=True,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=100,  # Adjust the logging steps to see more frequent updates
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

    # Only keep the mid fmeasure score for each metric
    result = {key: value['fmeasure'] * 100 if isinstance(value, dict) else value * 100 for key, value in result.items()}

    return result

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=eval_subset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate model on the test set
test_results = trainer.evaluate(eval_dataset=eval_subset)

# Print evaluation results
print(test_results)

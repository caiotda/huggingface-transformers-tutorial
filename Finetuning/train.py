import numpy as np

from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer,DataCollatorWithPadding, Trainer, TrainingArguments

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

checkpoint = "bert-base-uncased"
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

raw_datasets = load_dataset("glue", "mrpc")

"""
The first step is to define a training argument. That loads the hyperparameters the trainer will actually use for training and evaluation
"""

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

# We'll use a sequence classification model with two labels
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


"""
The default for the data collator is the data collator with padding, so we didn't need to pass it above. Now let's train.
"""
trainer.train()

from datasets import list_datasets
import tensorflow as tf
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score


emotions = load_dataset('emotion')

train_ds = emotions['train']

emotions.set_format(type = 'pandas')

df = emotions['train'][:]

def label_int2str(row):
    return emotions['train'].features['label'].int2str(row)
df['label_name'] = df['label'].apply(label_int2str)

emotions.reset_format()

model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched = True, batch_size = None)

num_labels = 6

tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
tf_model

tokenizer_columns = tokenizer.model_input_names

batch_size = 64

tf_train_dataset = emotions_encoded['train'].to_tf_dataset(columns = tokenizer_columns, 
                                                           label_cols = ['label'], 
                                                           shuffle=True, batch_size=batch_size)

tf_validation_dataset = emotions_encoded['validation'].to_tf_dataset(columns = tokenizer_columns, 
                                                           label_cols = ['label'], 
                                                           shuffle=True, batch_size=batch_size)


tf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
                 loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics = tf.metrics.SparseCategoricalAccuracy())

tf_model.fit(tf_train_dataset, validation_data = tf_validation_dataset, epochs = 2)




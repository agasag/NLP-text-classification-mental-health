from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

import tensorflow as tf
from transformers import TFBertModel, EarlyStoppingCallback, IntervalStrategy

class BertMultiLabelClassifier(tf.keras.Model):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.classifier = tf.keras.layers.Dense(num_labels, activation="sigmoid")

    def call(self, inputs):
        outputs = self.bert(inputs)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.classifier(x)
def preprocess_function(examples):
    """preprocessing function to tokenize text and truncate sequences
    to be no longer than DistilBERT’s maximum input length"""
    return tokenizer(examples["ori_text"], truncation=True)

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)

""" load hf datasets - train """
ds_cfs = load_dataset("Psychotherapy-LLM/CBT-Bench", "core_fine_test")
ds_cms = load_dataset("Psychotherapy-LLM/CBT-Bench", "core_major_test")
ds_ds = load_dataset("Psychotherapy-LLM/CBT-Bench", "distortions_test")

""" load hf datasets - test """
"""ds_cfs_test = load_dataset("Psychotherapy-LLM/CBT-Bench", "core_fine_test")
ds_cms_test = load_dataset("Psychotherapy-LLM/CBT-Bench", "core_major_test")
ds_ds_test = load_dataset("Psychotherapy-LLM/CBT-Bench", "distortions_test")"""


""" get targets & transform  to numerical labels """
core_beliefs_fine = ds_cfs['train']['core_belief_fine_grained']
core_beliefs_major = ds_cms['train']['core_belief_major']
distortions = ds_ds['train']['distortions']

mlb_core_beliefs_fine  = MultiLabelBinarizer()
mlb_core_beliefs_major  = MultiLabelBinarizer()
mlb_distortions  = MultiLabelBinarizer()

y_core_beliefs_fine = mlb_core_beliefs_fine.fit_transform(core_beliefs_fine)
label_to_index_core_beliefs_fine = {label: idx for idx, label in enumerate(mlb_core_beliefs_fine.classes_)}

y_core_beliefs_major = mlb_core_beliefs_major.fit_transform(core_beliefs_major)
label_to_index_core_beliefs_major = {label: idx for idx, label in enumerate(mlb_core_beliefs_major.classes_)}

y_distortions= mlb_distortions.fit_transform(distortions)
label_to_index_distortions = {label: idx for idx, label in enumerate(mlb_distortions.classes_)}

"""tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

tokenized_ds_cfs = ds_cfs.map(preprocess_function, batched=True)
tokenized_ds_cms = ds_cms.map(preprocess_function, batched=True)
tokenized_ds_ds = ds_ds.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")"""

""" """
#import tensorflow_hub as hub
X_train, X_test, y_train, y_test = train_test_split(ds_cfs['train']['ori_text'], y_core_beliefs_fine, test_size=0.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train_texts = tokenizer(X_train, padding=True, truncation=True, max_length=128, return_tensors="tf")

X_test_texts = tokenizer(X_test, padding=True, truncation=True, max_length=128, return_tensors="tf")


#bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
#bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

num_labels = y_train.shape[1]
model = BertMultiLabelClassifier("bert-base-uncased", num_labels)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.AUC(multi_label=True)]
)

history = model.fit(
    x={'input_ids': X_train_texts['input_ids'], 'attention_mask': X_train_texts['attention_mask']},
    y=y_train,
    batch_size=4,
    epochs=100,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

preds = model.predict({'input_ids': X_test_texts['input_ids'], 'attention_mask': X_test_texts['attention_mask']})
pred_text_labels = mlb_core_beliefs_fine.inverse_transform(preds > 0.5)
pred_num_labels = np.zeros(preds.shape)
pred_num_labels[preds > 0.5] = 1

# y_true: shape (num_samples, num_labels)
# y_pred: binary predictions after thresholding

# Common average options
f1 = f1_score(y_test, pred_num_labels, average='micro')   # good for imbalance
precision = precision_score(y_test, pred_num_labels, average='micro')
recall = recall_score(y_test, pred_num_labels, average='micro')

print(f"F1 (micro): {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

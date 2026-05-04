from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = ["I love AI", "I hate bugs"]
labels = [1, 0]

# Tokenize
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")

# Load model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=model.compute_loss,
    metrics=['accuracy']
)

# Train
model.fit(encodings['input_ids'], labels, epochs=2)

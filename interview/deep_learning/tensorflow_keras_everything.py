"""
TensorFlow + Keras — Everything
================================
pip install tensorflow
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, losses, metrics
print(f"TensorFlow version: {tf.__version__}")

# ════════════════════════════════════════════
# 1. BASIC NEURAL NETWORK
# ════════════════════════════════════════════
print("\n1. BASIC NEURAL NETWORK")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(20,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.summary()
# Total params: ~11,000

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.BinaryCrossentropy(),
    metrics=[metrics.AUC(name="auc"), metrics.Accuracy()]
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ],
    verbose=0
)

loss, auc, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test AUC: {auc:.4f}, Accuracy: {acc:.4f}")

# ════════════════════════════════════════════
# 2. CNN — Image Classification
# ════════════════════════════════════════════
print("\n2. CNN — IMAGE CLASSIFICATION (CIFAR-10)")

(X_train_img, y_train_img), (X_test_img, y_test_img) = keras.datasets.cifar10.load_data()
X_train_img = X_train_img.astype("float32") / 255.0   # normalize to [0,1]
X_test_img  = X_test_img.astype("float32") / 255.0
y_train_img = keras.utils.to_categorical(y_train_img, 10)
y_test_img  = keras.utils.to_categorical(y_test_img, 10)
print(f"Train: {X_train_img.shape}, Test: {X_test_img.shape}")
# Train: (50000, 32, 32, 3)

cnn = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(32,32,3)),
    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

cnn.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Data augmentation
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Train with augmentation
# history_cnn = cnn.fit(data_aug(X_train_img), y_train_img,
#                       epochs=50, batch_size=64, validation_split=0.1,
#                       callbacks=[callbacks.EarlyStopping(patience=10)])

print("CNN model built:", cnn.output_shape)  # (None, 10)

# ════════════════════════════════════════════
# 3. TRANSFER LEARNING
# ════════════════════════════════════════════
print("\n3. TRANSFER LEARNING — ResNet50")

base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # freeze pretrained weights

tl_model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

tl_model.compile(optimizer=optimizers.Adam(1e-4),
                 loss="categorical_crossentropy", metrics=["accuracy"])

print(f"Trainable params: {sum([np.prod(v.shape) for v in tl_model.trainable_variables]):,}")

# Fine-tuning: unfreeze top layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False  # keep bottom frozen

tl_model.compile(optimizer=optimizers.Adam(1e-5),  # very small LR!
                 loss="categorical_crossentropy", metrics=["accuracy"])
print("Fine-tuning setup done")

# ════════════════════════════════════════════
# 4. RNN / LSTM — Text Sentiment
# ════════════════════════════════════════════
print("\n4. LSTM — SENTIMENT ANALYSIS (IMDB)")

VOCAB_SIZE   = 10000
MAX_LEN      = 200
EMBED_DIM    = 64

(X_train_txt, y_train_txt), (X_test_txt, y_test_txt) = keras.datasets.imdb.load_data(
    num_words=VOCAB_SIZE
)
X_train_txt = keras.preprocessing.sequence.pad_sequences(X_train_txt, maxlen=MAX_LEN)
X_test_txt  = keras.preprocessing.sequence.pad_sequences(X_test_txt,  maxlen=MAX_LEN)
print(f"Train: {X_train_txt.shape}")  # (25000, 200)

lstm_model = keras.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

lstm_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
lstm_model.summary()

# ════════════════════════════════════════════
# 5. GRU
# ════════════════════════════════════════════
print("\n5. GRU — SEQUENCE MODEL")

gru_model = keras.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    layers.GRU(64, return_sequences=True),
    layers.GRU(32),
    layers.Dense(1, activation="sigmoid")
])
gru_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("GRU model built:", gru_model.output_shape)  # (None, 1)

# ════════════════════════════════════════════
# 6. CUSTOM TRAINING LOOP
# ════════════════════════════════════════════
print("\n6. CUSTOM TRAINING LOOP")

# Convert to tensors
X_tr_t = tf.constant(X_train, dtype=tf.float32)
y_tr_t = tf.constant(y_train, dtype=tf.float32)

simple_model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(20,)),
    layers.Dense(1, activation="sigmoid")
])

optimizer_custom = optimizers.Adam(1e-3)
loss_fn          = losses.BinaryCrossentropy()

@tf.function   # compile to graph for speed
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = simple_model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)

    gradients = tape.gradient(loss, simple_model.trainable_variables)
    optimizer_custom.apply_gradients(
        zip(gradients, simple_model.trainable_variables)
    )
    return loss

BATCH_SIZE = 32
dataset = tf.data.Dataset.from_tensor_slices((X_tr_t, y_tr_t))
dataset = dataset.shuffle(1000).batch(BATCH_SIZE)

for epoch in range(5):
    for x_batch, y_batch in dataset:
        loss = train_step(x_batch, y_batch)
    print(f"Epoch {epoch+1}/5 — Loss: {loss:.4f}")

# ════════════════════════════════════════════
# 7. FUNCTIONAL API (multi-input / multi-output)
# ════════════════════════════════════════════
print("\n7. FUNCTIONAL API")

# Multi-input model
text_input  = keras.Input(shape=(MAX_LEN,), name="text")
meta_input  = keras.Input(shape=(10,),     name="meta")

# Text branch
x = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(text_input)
x = layers.LSTM(32)(x)

# Meta branch
m = layers.Dense(16, activation="relu")(meta_input)

# Combine
combined = layers.concatenate([x, m])
out = layers.Dense(32, activation="relu")(combined)
out = layers.Dense(1, activation="sigmoid")(out)

multi_input_model = keras.Model(inputs=[text_input, meta_input], outputs=out)
print("Multi-input model:", multi_input_model.input_names)
# ['text', 'meta']

# ════════════════════════════════════════════
# 8. CALLBACKS
# ════════════════════════════════════════════
print("\n8. CALLBACKS")

cb_list = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=5, min_lr=1e-6, verbose=1
    ),
    callbacks.ModelCheckpoint(
        "best_model.h5", monitor="val_auc",
        save_best_only=True, mode="max", verbose=1
    ),
    callbacks.TensorBoard(log_dir="./logs"),
    callbacks.CSVLogger("training_log.csv"),
]
print(f"Callbacks ready: {[type(c).__name__ for c in cb_list]}")

# ════════════════════════════════════════════
# 9. SAVE & LOAD
# ════════════════════════════════════════════
print("\n9. SAVE AND LOAD")

model.save("model.keras")          # new format
# model.save("model.h5")           # old HDF5 format

loaded = keras.models.load_model("model.keras")
preds  = (loaded.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
from sklearn.metrics import accuracy_score
print(f"Loaded model accuracy: {accuracy_score(y_test, preds):.4f}")

# Save weights only
model.save_weights("weights.h5")
model.load_weights("weights.h5")

print("\nAll done! ✓")

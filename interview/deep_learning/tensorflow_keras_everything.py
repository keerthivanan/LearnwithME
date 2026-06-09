"""
TensorFlow + Keras — Definitions + Code + Outputs
===================================================
pip install tensorflow scikit-learn
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, losses, metrics
print(f"TensorFlow version: {tf.__version__}")


# ══════════════════════════════════════════════════════
# 1. BASIC NEURAL NETWORK (Dense / Fully Connected)
# ══════════════════════════════════════════════════════
# WHAT IS A NEURAL NETWORK?
#   → A model inspired by the brain: layers of "neurons" connected by weights
#   → Each neuron: output = activation(weights · inputs + bias)
#   → "Deep" learning = many layers → can learn complex patterns
#
# WHAT IS KERAS?
#   → High-level API on top of TensorFlow — makes building models easy
#   → Sequential API: stack layers one after another (most common)
#   → Functional API: for complex architectures (multi-input, branches)
#
# KEY LAYERS:
#   → Dense(units, activation) : fully connected layer — every neuron connects to every input
#   → BatchNormalization()     : normalize activations → faster training, more stable
#   → Dropout(rate)            : randomly zero out `rate` fraction of neurons → prevents overfitting
#
# ACTIVATIONS:
#   → relu    : max(0, x) → hides negative values → most common for hidden layers
#   → sigmoid : 1/(1+e^-x) → squashes to [0,1] → binary classification output
#   → softmax : probability distribution across N classes → multi-class output
#
# COMPILE PARAMS:
#   → optimizer : HOW weights are updated (Adam is usually best)
#   → loss      : WHAT to minimize (BinaryCrossentropy for binary, CategoricalCrossentropy for multi-class)
#   → metrics   : what to MONITOR (not minimized, just shown)

print("\n" + "=" * 55)
print("1. BASIC NEURAL NETWORK")
print("=" * 55)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler  = StandardScaler()
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
    layers.Dense(1, activation="sigmoid")   # binary output
])

model.summary()
# Total params: ~11,000

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.BinaryCrossentropy(),
    metrics=[metrics.AUC(name="auc")]
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

loss_val, auc_val = model.evaluate(X_test, y_test, verbose=0)
print(f"Test AUC: {auc_val:.4f}")   # e.g. Test AUC: 0.9234


# ══════════════════════════════════════════════════════
# 2. CNN — Convolutional Neural Network (Image Classification)
# ══════════════════════════════════════════════════════
# WHAT IS A CNN?
#   → A neural network designed for IMAGES (grids of pixels)
#   → Instead of connecting to ALL pixels, it uses local filters (kernels)
#   → Learns to detect: edges → shapes → textures → objects (bottom to top)
#
# KEY LAYERS:
#   → Conv2D(filters, kernel_size, activation, padding)
#       - filters     : how many feature detectors to learn (32, 64, 128...)
#       - kernel_size : size of the filter (3x3 is most common)
#       - padding="same": keep spatial dimensions the same
#   → MaxPooling2D(2,2) : downsample by taking max in each 2x2 region
#                         reduces spatial size by half → fewer params, more robust
#   → Flatten()         : convert 3D feature maps → 1D vector for Dense layers
#
# DATA AUGMENTATION:
#   → Randomly flip/rotate/zoom images during training
#   → Model sees more variety → better generalization → less overfitting
#   → Only apply to TRAINING data, never test data!
#
# CIFAR-10: 60,000 images (32×32 RGB) in 10 classes (airplane, car, bird, etc.)

print("\n" + "=" * 55)
print("2. CNN — IMAGE CLASSIFICATION (CIFAR-10)")
print("=" * 55)

(X_train_img, y_train_img), (X_test_img, y_test_img) = keras.datasets.cifar10.load_data()
X_train_img = X_train_img.astype("float32") / 255.0   # normalize to [0,1]
X_test_img  = X_test_img.astype("float32")  / 255.0
y_train_img = keras.utils.to_categorical(y_train_img, 10)   # one-hot encode
y_test_img  = keras.utils.to_categorical(y_test_img,  10)
print(f"Train: {X_train_img.shape}, Test: {X_test_img.shape}")
# Train: (50000, 32, 32, 3)

cnn = keras.Sequential([
    # Block 1 — detect basic edges
    layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(32,32,3)),
    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),   # 32x32 → 16x16
    layers.Dropout(0.25),

    # Block 2 — detect shapes
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),   # 16x16 → 8x8
    layers.Dropout(0.25),

    # Block 3 — detect textures/objects
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),   # 8x8 → 4x4
    layers.Dropout(0.25),

    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")   # 10 classes
])

cnn.compile(optimizer=optimizers.Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"])

# Data augmentation layer
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

print("CNN model built:", cnn.output_shape)   # (None, 10)
# To train: cnn.fit(data_aug(X_train_img), y_train_img, epochs=50, ...)


# ══════════════════════════════════════════════════════
# 3. TRANSFER LEARNING
# ══════════════════════════════════════════════════════
# WHAT IS TRANSFER LEARNING?
#   → Use a model pre-trained on a LARGE dataset (ImageNet: 1.4M images, 1000 classes)
#   → "Transfer" what it learned to your smaller, specific dataset
#   → You don't train from scratch — use the knowledge it already has!
#
# PROCESS:
#   1. Load pre-trained base model (ResNet50, VGG16, EfficientNet...)
#   2. Freeze base weights: base_model.trainable = False
#   3. Add your own classification head (Dense layers)
#   4. Train ONLY the head (fast, few params)
#   5. Optional: unfreeze top layers of base → fine-tune with very small LR
#
# WHY FREEZE BASE FIRST?
#   → If you update all weights at once, your random head will destroy
#     the carefully learned features in the base model
#   → Train head first → then gradually unfreeze for fine-tuning
#
# GlobalAveragePooling2D:
#   → Compresses feature maps → single vector (better than Flatten for CNNs)

print("\n" + "=" * 55)
print("3. TRANSFER LEARNING — ResNet50")
print("=" * 55)

base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,       # remove ImageNet's final Dense layer
    input_shape=(224, 224, 3)
)
base_model.trainable = False   # Step 1: freeze all base weights

tl_model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

tl_model.compile(optimizer=optimizers.Adam(1e-3),
                 loss="categorical_crossentropy", metrics=["accuracy"])
print(f"Trainable params (head only): {sum([np.prod(v.shape) for v in tl_model.trainable_variables]):,}")

# Step 5 — fine-tune: unfreeze top 20 layers with tiny LR
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

tl_model.compile(optimizer=optimizers.Adam(1e-5),   # very small LR — don't destroy learned features
                 loss="categorical_crossentropy", metrics=["accuracy"])
print("Fine-tuning setup done — LR dropped to 1e-5")


# ══════════════════════════════════════════════════════
# 4. LSTM — Long Short-Term Memory (Text / Sequences)
# ══════════════════════════════════════════════════════
# WHAT IS AN RNN?
#   → Recurrent Neural Network — processes sequences one step at a time
#   → Has a "hidden state" that carries memory from previous steps
#   → Problem: vanilla RNN forgets long-range dependencies
#
# WHAT IS LSTM?
#   → Long Short-Term Memory — solves RNN's forgetting problem
#   → Uses 3 GATES to control what to remember/forget/output:
#     - Forget gate : "should I erase this memory?"
#     - Input gate  : "should I add new info to memory?"
#     - Output gate : "what should I output right now?"
#   → Can learn dependencies across 100s of tokens
#
# WHAT IS BIDIRECTIONAL LSTM?
#   → Runs LSTM in BOTH directions (left→right AND right→left)
#   → Doubles the information — sees full context from both sides
#   → Great for text understanding (not generation)
#
# EMBEDDING LAYER:
#   → Converts integer word IDs → dense vectors (word embeddings)
#   → Learns that similar words should have similar vectors
#   → Embedding(vocab_size, embed_dim) → output shape: (batch, seq_len, embed_dim)

print("\n" + "=" * 55)
print("4. LSTM — SENTIMENT ANALYSIS (IMDB)")
print("=" * 55)

VOCAB_SIZE = 10000
MAX_LEN    = 200
EMBED_DIM  = 64

(X_train_txt, y_train_txt), (X_test_txt, y_test_txt) = keras.datasets.imdb.load_data(
    num_words=VOCAB_SIZE
)
X_train_txt = keras.preprocessing.sequence.pad_sequences(X_train_txt, maxlen=MAX_LEN)
X_test_txt  = keras.preprocessing.sequence.pad_sequences(X_test_txt,  maxlen=MAX_LEN)
print(f"Train shape: {X_train_txt.shape}")   # (25000, 200)

lstm_model = keras.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),   # return_sequences=True → pass all steps to next layer
    layers.Bidirectional(layers.LSTM(32)),                           # return_sequences=False → only last step
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
lstm_model.summary()
# Trainable params: ~2M


# ══════════════════════════════════════════════════════
# 5. GRU — Gated Recurrent Unit
# ══════════════════════════════════════════════════════
# WHAT IS GRU?
#   → Simplified version of LSTM with only 2 gates (vs LSTM's 3)
#   → Reset gate  : how much past info to forget
#   → Update gate : blend of old and new state
#   → Faster and fewer params than LSTM, often similar accuracy
#   → Choose GRU when: speed matters and sequences aren't super long
#   → Choose LSTM when: need maximum accuracy on long sequences

print("\n" + "=" * 55)
print("5. GRU — SEQUENCE MODEL")
print("=" * 55)

gru_model = keras.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    layers.GRU(64, return_sequences=True),
    layers.GRU(32),
    layers.Dense(1, activation="sigmoid")
])
gru_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("GRU model output shape:", gru_model.output_shape)   # (None, 1)


# ══════════════════════════════════════════════════════
# 6. CUSTOM TRAINING LOOP
# ══════════════════════════════════════════════════════
# WHAT IS A CUSTOM TRAINING LOOP?
#   → Instead of model.fit(), you manually control each training step
#   → Gives FULL flexibility: custom loss, multiple optimizers, complex logic
#
# KEY CONCEPTS:
#   → GradientTape  : records all operations to compute gradients automatically
#     - tape.gradient(loss, variables) → computes ∂loss/∂variable for each variable
#   → apply_gradients : update weights: weight = weight - lr * gradient
#   → @tf.function  : compile the function to a TF graph → 2-10x faster
#   → tf.data.Dataset: efficient data pipeline with shuffle + batch

print("\n" + "=" * 55)
print("6. CUSTOM TRAINING LOOP")
print("=" * 55)

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
        loss        = loss_fn(y_batch, predictions)
    gradients = tape.gradient(loss, simple_model.trainable_variables)
    optimizer_custom.apply_gradients(zip(gradients, simple_model.trainable_variables))
    return loss

dataset = tf.data.Dataset.from_tensor_slices((X_tr_t, y_tr_t))
dataset = dataset.shuffle(1000).batch(32)

for epoch in range(5):
    for x_batch, y_batch in dataset:
        loss = train_step(x_batch, y_batch)
    print(f"  Epoch {epoch+1}/5 — Loss: {loss:.4f}")
# Epoch 1/5 — Loss: 0.4821
# Epoch 5/5 — Loss: 0.2134


# ══════════════════════════════════════════════════════
# 7. FUNCTIONAL API — Multi-Input / Multi-Output Models
# ══════════════════════════════════════════════════════
# WHAT IS THE FUNCTIONAL API?
#   → Alternative to Sequential API for complex architectures
#   → Sequential: only one input → one output, strictly linear
#   → Functional: multiple inputs, branches, skip connections, multiple outputs
#
# WHEN TO USE FUNCTIONAL API:
#   → Multi-input models (text + metadata, image + tabular)
#   → Multi-output models (predict age + gender + location)
#   → Residual connections (ResNet-style skip connections)
#   → Shared layers (same layer processes different inputs)
#
# PATTERN:
#   → inputs = keras.Input(shape=(...))  — define input shape
#   → x = layers.SomeLayer()(inputs)     — call layers like functions
#   → model = keras.Model(inputs=..., outputs=...)

print("\n" + "=" * 55)
print("7. FUNCTIONAL API")
print("=" * 55)

text_input = keras.Input(shape=(MAX_LEN,), name="text")
meta_input = keras.Input(shape=(10,),     name="meta")

# Text branch — LSTM for sequence data
x = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(text_input)
x = layers.LSTM(32)(x)

# Meta branch — Dense for tabular features
m = layers.Dense(16, activation="relu")(meta_input)

# Merge branches — concatenate both representations
combined = layers.concatenate([x, m])
out = layers.Dense(32, activation="relu")(combined)
out = layers.Dense(1,  activation="sigmoid")(out)

multi_model = keras.Model(inputs=[text_input, meta_input], outputs=out)
print("Multi-input model inputs:", multi_model.input_names)   # ['text', 'meta']
print("Output shape:", multi_model.output_shape)              # (None, 1)


# ══════════════════════════════════════════════════════
# 8. CALLBACKS — Control Training Behavior
# ══════════════════════════════════════════════════════
# WHAT ARE CALLBACKS?
#   → Functions that run at specific points during training
#   → Let you: stop early, save best model, reduce LR, log metrics
#
# MOST IMPORTANT CALLBACKS:
#   → EarlyStopping       : stop if val_loss doesn't improve for `patience` epochs
#                           restore_best_weights=True → revert to best checkpoint
#   → ReduceLROnPlateau   : if val_loss plateaus, reduce LR by `factor`
#                           plateau = learning rate is too large for current loss landscape
#   → ModelCheckpoint     : save model when val metric improves
#   → TensorBoard         : log metrics → visualize in TensorBoard UI
#   → CSVLogger           : write epoch metrics to CSV file

print("\n" + "=" * 55)
print("8. CALLBACKS")
print("=" * 55)

cb_list = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
    callbacks.ModelCheckpoint(
        "best_model.keras", monitor="val_auc",
        save_best_only=True, mode="max", verbose=1
    ),
    callbacks.TensorBoard(log_dir="./logs"),
    callbacks.CSVLogger("training_log.csv"),
]
print(f"Callbacks ready: {[type(c).__name__ for c in cb_list]}")


# ══════════════════════════════════════════════════════
# 9. SAVE & LOAD MODELS
# ══════════════════════════════════════════════════════
# WHAT IS MODEL SAVING?
#   → Persist trained model to disk so you can use it later
#   → Two things to save: ARCHITECTURE (layers) + WEIGHTS (learned values)
#
# FORMATS:
#   → .keras    : new native format (recommended) — saves everything
#   → .h5       : HDF5 format (legacy) — also saves everything
#   → SavedModel: TensorFlow format — for deployment (TFServing, TFLite)
#
# WEIGHTS ONLY:
#   → save_weights() / load_weights() — save ONLY the numbers, not architecture
#   → Useful for: checkpointing mid-training, transfer learning

print("\n" + "=" * 55)
print("9. SAVE & LOAD")
print("=" * 55)

model.save("model.keras")         # save architecture + weights + optimizer state
loaded = keras.models.load_model("model.keras")

preds = (loaded.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
from sklearn.metrics import accuracy_score
print(f"Loaded model accuracy: {accuracy_score(y_test, preds):.4f}")
# Loaded model accuracy: 0.9025

# Save/load weights only
model.save_weights("weights.weights.h5")
model.load_weights("weights.weights.h5")
print("Weights saved and reloaded successfully")

print("\nAll done! ✓")

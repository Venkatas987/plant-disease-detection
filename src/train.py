import tensorflow as tf
import json
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# ---------------- PATHS ----------------
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
LABEL_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# ---------------- PARAMETERS ----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5                 # 🔥 reduced epochs for speed
STEPS_PER_EPOCH = 300      # 🔥 limit steps
VAL_STEPS = 50             # 🔥 limit validation steps

os.makedirs(MODEL_DIR, exist_ok=True)

print("🚀 Starting training on full PlantVillage dataset...")

# ---------------- DATA GENERATORS ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = len(train_data.class_indices)
print("📊 Total classes detected:", num_classes)

# ---------------- SAVE CLASS LABELS ----------------
with open(LABEL_PATH, "w") as f:
    json.dump(train_data.class_indices, f)

# ---------------- MODEL ----------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(base_model.input, output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- CALLBACKS ----------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, save_best_only=True, monitor="val_accuracy"
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True
    )
]

# ---------------- TRAIN ----------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    callbacks=callbacks
)

print("✅ Training completed successfully!")
print("💾 Model saved at:", MODEL_PATH)
print("📁 Class labels saved at:", LABEL_PATH)

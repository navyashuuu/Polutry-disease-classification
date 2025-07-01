import os
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ✅ Paths
zip_path = "/content/ncd.zip"          # Replace with your actual ZIP path
csv_path = "/content/Polutry.csv"
image_dir = "/content"

# ✅ 1. Extract ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(image_dir)

# ✅ 2. Load CSV
df = pd.read_csv(csv_path)

# ✅ 3. Map images from ZIP
all_images = []
for root, dirs, files in os.walk(image_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append(os.path.join(root, f))

# ✅ 4. Map filename → full path
image_map = {os.path.basename(p): p for p in all_images}
df['images'] = df['images'].map(image_map)
df = df.dropna(subset=['images'])

# ✅ 5. Check if data exists
if df.empty:
    raise ValueError("❌ No matching images found. Please verify the CSV and ZIP contents.")

# ✅ 6. Label encode poultry disease names
lb = LabelBinarizer()
labels = lb.fit_transform(df['label'])
class_names = lb.classes_
print("📋 Poultry Disease Classes:", list(class_names))

# ✅ 6.5 Show sample poultry disease images
sample_df = df.sample(8)
plt.figure(figsize=(14, 6))
for idx, row in enumerate(sample_df.itertuples()):
    img = load_img(row.images, target_size=(224, 224))
    plt.subplot(2, 4, idx + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Disease: {row.label}")
plt.suptitle("📸 Sample Poultry Disease Images", fontsize=16)
plt.tight_layout()
plt.show()

# ✅ 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['images'], labels, test_size=0.2, random_state=42)

# ✅ 8. Image preprocessing
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

X_train_array = np.array([preprocess_image(p) for p in X_train])
X_test_array = np.array([preprocess_image(p) for p in X_test])

# ✅ 9. Build MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(labels.shape[1], activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ 10. Train the model
history = model.fit(X_train_array, y_train, validation_data=(X_test_array, y_test), epochs=10, batch_size=32)

# ✅ 11. Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Poultry Disease Classification Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ✅ 12. Predict on test set
preds = model.predict(X_test_array)
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(y_test, axis=1)

# ✅ 13. Show sample predictions
print("\n🔍 Sample Predictions:")
for i in range(5):
    print(f"True: {class_names[y_true[i]]} | Predicted: {class_names[y_pred[i]]}")

# ✅ 14. Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

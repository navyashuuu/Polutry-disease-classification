import os
import zipfile
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# ==== File Paths ====

csv_path = r"C:\Users\Navya\Downloads\train_data.csv"
zip_path = r"C:\Users\Navya\Downloads\archive.zip"
image_dir = "images"
model_path = "model/poultry_model.h5"

# ==== Extract ZIP Safely ====
if not os.path.exists(image_dir):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"❌ ZIP file not found at path: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(f"❌ The file at '{zip_path}' is not a valid ZIP file.")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(image_dir)
        print("✅ ZIP Extracted.")
    except zipfile.BadZipFile:
        raise zipfile.BadZipFile("❌ Corrupted ZIP file. Please re-download or check the file integrity.")
    except Exception as e:
        raise RuntimeError(f"❌ Unexpected error during ZIP extraction: {e}")
else:
    print("✅ ZIP Already Extracted.")

# ==== Load CSV ====
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ CSV file not found at path: {csv_path}")

df = pd.read_csv(csv_path)
df.dropna(inplace=True)
print("✅ CSV Columns:", df.columns.tolist())
print("🔍 Sample rows:\n", df.head())

# ==== Detect filename column ====
filename_col = None
for col in df.columns:
    if "file" in col.lower() or "image" in col.lower():
        filename_col = col
        break

if filename_col is None:
    raise ValueError("❌ No column found for filenames (e.g. 'filename' or 'image_name') in CSV!")

# ==== Image Preprocessing ====
data, labels = [], []
image_size = (224, 224)

for _, row in df.iterrows():
    try:
        img_file = row[filename_col]
        label = row["label"]
        img_path = os.path.join(image_dir, img_file)
        print(f"🖼️ Loading image: {img_path} with label: {label}")
        img = image.load_img(img_path, target_size=image_size)
        img = image.img_to_array(img)
        img = preprocess_input(img)
        data.append(img)
        labels.append(label)
    except Exception as e:
        print(f"⚠️ Error loading image {row[filename_col]}: {e}")

if len(data) == 0 or len(labels) == 0:
    raise ValueError("❌ No valid images or labels found. Please check CSV and image folder.")

data = np.array(data)
labels = np.array(labels)

# ==== Encode Labels ====
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(f"✅ Encoded classes: {lb.classes_}")

# ==== Train/Test Split ====
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# ==== Load or Train Model ====
if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ Model loaded from disk.")
else:
    print("⚙️ Training new model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(lb.classes_), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    os.makedirs("model", exist_ok=True)
    model.save(model_path)
    print("✅ Model trained and saved.")

# ==== Flask App ====
app = Flask(__name__)
os.makedirs("static", exist_ok=True)  # Ensure static folder exists

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return "No file uploaded", 400

        file = request.files['file']
        filename = file.filename
        save_path = os.path.join("static", filename)
        file.save(save_path)

        try:
            img = image.load_img(save_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            predicted_class = lb.classes_[np.argmax(prediction)]
            return render_template("predict.html", prediction=predicted_class, image_file=filename)
        except Exception as e:
            return f"Prediction failed: {str(e)}", 500

    return '''
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <input type="submit" value="Predict">
        </form>
    '''

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    mobile = request.form.get("mobile")
    return f"<h3>Thank you! We'll contact you at {mobile}.</h3>"

if __name__ == '__main__':
    app.run(debug=True)

import os
import io
import json
import base64
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'skin_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'class_labels.json')

print("Loading model, please wait...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)
print("Model loaded successfully.")

DISPLAY_NAMES = {
    'Acne and Rosacea Photos': 'Acne and Rosacea',
    'Eczema Photos': 'Eczema',
    'Nail Fungus and other Nail Disease': 'Nail Fungus and other Nail Diseases',
    'Psoriasis pictures Lichen Planus and related diseases': 'Psoriasis / Lichen Planus',
    'Seborrheic Keratoses and other Benign Tumors': 'Seborrheic Keratoses / Benign Tumors',
    'Tinea Ringworm Candidiasis and other Fungal Infections': 'Tinea / Ringworm / Fungal Infections',
    'Warts Molluscum and other Viral Infections': 'Warts / Molluscum / Viral Infections',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': 'Actinic Keratosis / Basal Cell Carcinoma / Malignant Lesions',
    'Melanoma Skin Cancer Nevi and Moles': 'Melanoma / Skin Cancer / Nevi and Moles',
    'Light Diseases and Disorders of Pigmentation': 'Light-related and Pigmentation Disorders',
}

# Classes that may involve skin cancer - shown with a "see a doctor soon" notice
URGENT_CLASSES = {
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Melanoma Skin Cancer Nevi and Moles',
}

def pretty(name):
    return DISPLAY_NAMES.get(name, name)

# Locate the nested MobileNetV2 base inside the model (needed for Grad-CAM)
_base_idx = None
for _i, _layer in enumerate(model.layers):
    if isinstance(_layer, tf.keras.Model):
        _base_idx = _i
        break


def _make_gradcam(arr, class_index, img_uint8):
    """Returns a base64 PNG data-URI with a soft attention overlay, or raises on failure."""
    if _base_idx is None:
        raise RuntimeError("Base model layer not found")

    pre_layers = model.layers[1:_base_idx]       # e.g. Rescaling
    base = model.layers[_base_idx]
    head_layers = model.layers[_base_idx + 1:]   # pooling, dropout, dense...

    x = tf.convert_to_tensor(arr, dtype=tf.float32)
    with tf.GradientTape() as tape:
        h = x
        for layer in pre_layers:
            h = layer(h)
        conv = base(h, training=False)           # last conv feature map (7x7x1280)
        tape.watch(conv)
        y = conv
        for layer in head_layers:
            y = layer(y)
        score = y[:, class_index]

    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    cam = cam / (cam.max() + 1e-8)

    # Smooth upsampling so the map is not blocky
    heat = Image.fromarray(np.uint8(cam * 255)).resize((224, 224), Image.BICUBIC)
    heat = heat.filter(ImageFilter.GaussianBlur(radius=8))
    hm = np.asarray(heat).astype(np.float32) / 255.0
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

    # Show colour only where attention is strong; weak areas stay as the original photo
    strength = np.clip((hm - 0.40) / 0.60, 0, 1)

    # Soft warm colour ramp: yellow -> orange -> red
    r = np.full_like(hm, 255.0)
    g = np.clip(230 - 190 * hm, 40, 230)
    b = np.full_like(hm, 40.0)
    colored = np.stack([r, g, b], axis=-1)

    alpha = (strength * 0.60)[..., None]
    blended = img_uint8.astype(np.float32) * (1 - alpha) + colored * alpha

    out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def predict_image(image_source):
    """Accepts a file object or a file path. Returns a result dictionary."""
    img = Image.open(image_source).convert('RGB').resize((224, 224))
    img_uint8 = np.array(img)
    arr = np.expand_dims(img_uint8 / 255.0, axis=0).astype(np.float32)

    preds = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:3]
    best = int(top_idx[0])
    raw_name = class_labels[str(best)]

    # Urgent if the top prediction OR a close alternative is a cancer-related class
    urgent = raw_name in URGENT_CLASSES or any(
        class_labels[str(int(i))] in URGENT_CLASSES and preds[i] >= 0.20
        for i in top_idx
    )

    result = {
        'predicted_class': pretty(raw_name),
        'confidence': float(preds[best]),
        'urgent': urgent,
        'top3': [
            {'name': pretty(class_labels[str(int(i))]),
             'confidence': round(float(preds[i]) * 100, 1)}
            for i in top_idx
        ],
        'gradcam': None,
    }

     # Grad-CAM disabled (heatmaps did not align reliably with lesions on this dataset)

    return result
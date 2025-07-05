import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

IMG_SIZE = (224, 224)
THRESHOLD = 0.80

# โหลดโมเดลแค่ครั้งเดียว
embedding_model = EfficientNetV2S(include_top=False, weights="imagenet", pooling="avg")

def load_image_embedding(image):
    if isinstance(image, str):
        img = Image.open(image).convert("RGB").resize(IMG_SIZE)
    else:
        image.seek(0)
        img = Image.open(image).convert("RGB").resize(IMG_SIZE)

    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    embedding = embedding_model.predict(img_array, verbose=0)
    return embedding[0]

def is_mango_leaf_from_embedding(image, reference_embeddings, threshold=THRESHOLD):
    if len(reference_embeddings) == 0:
        raise ValueError("Reference embeddings ว่างเปล่า")
    query_embedding = load_image_embedding(image)
    sims = cosine_similarity([query_embedding], reference_embeddings)[0]
    max_sim = np.max(sims)
    return max_sim >= threshold, max_sim

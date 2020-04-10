import argparse
import tensorflow as tf
import json
import numpy as np
from PIL import Image
import tensorflow_hub as hub


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="image path")
ap.add_argument("-m", "--model", required=True,
	help="model path")
ap.add_argument("-k", "--top_k", required=False,
	help="top_k probs of the image")
ap.add_argument("-c", "--category_names", required=False,
	help="classes")
args = vars(ap.parse_args())

image_path = args['image']
saved_keras_model_filepath = args['model']
top_k = args['top_k']
category_names = args['category_names']
image_size = 224


reloaded_keras_model = tf.keras.experimental.load_from_saved_model(
    saved_keras_model_filepath, custom_objects={'KerasLayer': hub.KerasLayer})

reloaded_keras_model.summary()


def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()


def predict_class(image_path, model, top_k=0):
    if top_k < 1:
        top_k = 1
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    expanded_image = np.expand_dims(image, axis=0)
    probes = model.predict(expanded_image)
    top_k_values, top_k_indices = tf.nn.top_k(probes, k=top_k)

    top_k_values = top_k_values.numpy()
    top_k_indices = top_k_indices.numpy()

    return top_k_values, top_k_indices


top_k_values, top_k_indices = predict_class(image_path, reloaded_keras_model, top_k=int(top_k))


print('Propabilties:', top_k_values)
print('Classes Keys:', top_k_indices)

if category_names != None:
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    print("Classes Values:")
    for idx in top_k_indices[0]:
        print("-",class_names[str(idx+1)])

    
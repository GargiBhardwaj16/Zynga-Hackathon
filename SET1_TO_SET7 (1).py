#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model

def build_cnn(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    cnn_model = Model(inputs=input_img, outputs=x)
    return cnn_model
input_shape = (224, 224, 3)  

cnn_model = build_cnn(input_shape)
cnn_model.summary()


# In[2]:


def extract_features(model, img):
    features = model.predict(img)
    return features.flatten()


# In[10]:


def extract_text_from_image(img_path, keyword):
    results = reader.readtext(img_path, paragraph=True)
    for result in results:
        text = result[1]
        if keyword in text:
            return text.split()[-1]
    return None


# In[12]:


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import easyocr

reader = easyocr.Reader(['en'])

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def load_images_from_set(set_path):
    main_image_path = os.path.join(set_path, 'Image.png')
    test1_path = os.path.join(set_path, 'Test1.png')
    test2_path = os.path.join(set_path, 'Test2.png')
    
    main_image = load_and_preprocess_image(main_image_path)
    test1_image = load_and_preprocess_image(test1_path)
    test2_image = load_and_preprocess_image(test2_path)
    
    return main_image, test1_image, test2_image

def extract_total_win(results):
    for result in results:
        text = result[1]
        if 'TOTAL WIN' in text:
            parts = text.split()
            if parts and parts[0].replace(',', '').isdigit():
                return parts[0]
    return None

vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def get_similarity_scores(main_img, test1_img, test2_img):
    main_features = extract_features(vgg_model, main_img)
    test1_features = extract_features(vgg_model, test1_img)
    test2_features = extract_features(vgg_model, test2_img)
    
    sim1 = cosine_similarity([main_features], [test1_features])[0][0]
    sim2 = cosine_similarity([main_features], [test2_features])[0][0]
    
    return [sim1, sim2]

# OCR for text extraction
def extract_total_win(results):
    for result in results:
        text = result[1]
        if 'TOTAL WIN' in text:
            parts = text.split()
            if parts and parts[0].replace(',', '').isdigit():
                return parts[0]
    return None

def process_set(set_number, set_path):
    if set_number in range(1, 8):
        main_img, test1_img, test2_img = load_images_from_set(set_path)
        similarity_scores = get_similarity_scores(main_img, test1_img, test2_img)
        return similarity_scores
    
    elif set_number == 8:
        test1_path = os.path.join(set_path, 'Test1.png')
        test2_path = os.path.join(set_path, 'Test2.png')
        results1 = reader.readtext(test1_path, paragraph=True)
        results2 = reader.readtext(test2_path, paragraph=True)
        total_win_amount_test1 = extract_total_win(results1)
        total_win_amount_test2 = extract_total_win(results2)
        return [total_win_amount_test1, total_win_amount_test2]
    
def main():
    base_path = r'C:\Users\dushy\Downloads\Hackathon2024 (1)\SlotsProject\Problems' 
    results = {}

    for set_number in range(1, 9):
        set_path = os.path.join(base_path, f'Set{set_number}')
        result = process_set(set_number, set_path)
        results[f'Set{set_number}'] = result

    return results

if __name__ == "__main__":
    results = main()
    print(results)


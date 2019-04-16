import cv2
import tensorflow as tf

CATEGORIES = ["IEEE", "Macmillan", "SpringerNature", "WoltersKluwerHealth"]


def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x4-CNN.model")


prediction = model.predict([prepare('IEEEthumbnail.jpg')])
prediction1 = model.predict([prepare('Macmillanthumbnail.jpg')])
prediction2 = model.predict([prepare('SpringerNature.jpg')])
prediction3 = model.predict([prepare('dog.jpg')])
print(CATEGORIES[int(prediction[0][0])])
print(CATEGORIES[int(prediction1[0][0])])
print(CATEGORIES[int(prediction2[0][0])])
print(CATEGORIES[int(prediction3[0][0])])

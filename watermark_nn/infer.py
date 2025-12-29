import tensorflow as tf
import numpy as np
from PIL import Image

classes = ["cloud", "list", "umbrella"]

model = tf.keras.models.load_model("watermark_model.h5")


img_path = r"C:\Users\tvoro\OneDrive\Рабочий стол\курс\data\test\list\7.jpg"


try:
    img = Image.open(img_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Изображение не найдено: {img_path}")
except Exception as e:
    raise Exception(f"Ошибка при открытии изображения: {e}")


img = img.resize((320, 240))


img_array = np.array(img) / 255.0


img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions[0])
predicted_class = classes[predicted_class_idx]
confidence = predictions[0][predicted_class_idx]

print(f"Предсказанный класс: {predicted_class}")
print(f"Уверенность: {confidence:.4f}")
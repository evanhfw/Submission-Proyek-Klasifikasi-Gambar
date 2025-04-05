import tensorflow as tf
from tensorflow import keras

import cv2
import numpy as np

class CNNImageClassifier:
    def __init__(self):
        self.model = keras.models.load_model('best_model.keras')
        self.input_size = self.model.input_shape[1:3]
        
        self.idx_to_label = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
            19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del',
            27: 'nothing', 28: 'space'
        }
        
    def __call__(self, image_path):
        """
        Menerima array gambar (1, H, W, 3), mengeluarkan prediksi kelas.
        """
        img_array = self.load_image(image_path)
        preds = self.model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0]
        pred_label = self.idx_to_label[pred_class]
        return pred_class, pred_label, preds[0]
        
    def load_image(self, image_data):
        """
        Load dan resize gambar agar sesuai input model.
        Output: np.array dengan shape (1, height, width, channels)
        """
        if isinstance(image_data, str):
            # Jika input adalah path file
            img = cv2.imread(image_data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Jika input adalah data bytes dari widget upload
            img_bytes = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img = cv2.resize(img, self.input_size)
        img = np.expand_dims(img, axis=0)  # (1, H, W, 3)
        return img
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import numpy as np

# MNIST test verisini yükle
(_, _), (x_test, y_test) = datasets.mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Eğitilmiş modeli yükle
model = load_model('mnist_cnn_model.h5')
print("Model dosyadan yüklendi.")

# Rastgele 10 test örneği seç
num_examples = 10
random_indices = np.random.choice(len(x_test), num_examples, replace=False)
random_images = x_test[random_indices]
random_labels = y_test[random_indices]

# Tahmin yap
predictions = model.predict(random_images)

# Sonuçları çiz
plt.figure(figsize=(15, 5))
for i in range(num_examples):
    plt.subplot(2, 5, i+1)
    plt.imshow(random_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Gerçek: {random_labels[i]}\nTahmin: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.suptitle("Yüklenen Model İle Tahminler")
plt.show()

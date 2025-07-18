import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 1. Veri Kümesini Yükle ve Hazırla (MNIST)
# MNIST veri kümesi, Keras ile kolayca yüklenebilir
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Görsel boyutlarını CNN için uygun hale getir (kanal ekle)
# MNIST görselleri 28x28, ama CNN 28x28x1 (gri tonlamalı) veya 28x28x3 (renkli) ister.
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Piksel değerlerini normalize et (0-255 aralığından 0-1 aralığına)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(f"Eğitim verisi boyutu: {x_train.shape}")
print(f"Test verisi boyutu: {x_test.shape}")
print(f"Eğitim etiketleri boyutu: {y_train.shape}")
print(f"Test etiketleri boyutu: {y_test.shape}")

# 2. Basit Bir CNN Modeli Oluştur
model = keras.Sequential([
    # Evrişim Katmanı (Convolutional Layer)
    # Filtre sayısı: 32, Filtre boyutu: 3x3, Aktivasyon: ReLU
    # input_shape: Giriş görselinin boyutu (28x28x1 - yükseklik, genişlik, kanal sayısı)
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    
    # Havuzlama Katmanı (Pooling Layer - Max Pooling)
    # Pencere boyutu: 2x2. Bu, çıktıyı yarıya indirir (26x26 -> 13x13)
    layers.MaxPooling2D((2, 2)),
    
    # Bir Evrişim ve Havuzlama Katmanı Daha (Özellikleri daha da soyutlamak için)
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    
    # Evrişimli çıktıları Tam Bağlantılı katmanlara beslemek için düzleştir (Flatten)
    # 2D özellik haritalarını 1D vektöre dönüştürür
    layers.Flatten(),
    
    # Tam Bağlantılı Katman (Fully Connected Layer)
    # 64 nöronlu gizli katman. Görüntüdeki yüksek seviyeli özellikleri işler.
    layers.Dense(64, activation="relu"),
    
    # Çıkış Katmanı (Output Layer)
    # 10 nöron (0-9 arası rakamlar için). Softmax, olasılık dağılımı verir.
    layers.Dense(10, activation="softmax") 
])

# Modelin yapısını özetle
model.summary()

# 3. Modeli Derle (Compile)
# Optimizer: Modelin ağırlıklarını nasıl güncelleyeceğini belirler (Adam iyi bir varsayılan)
# Loss: Modelin tahminleri ile gerçek etiketler arasındaki farkı ölçer (sparse_categorical_crossentropy rakam sınıflandırması için iyi)
# Metrics: Eğitim ve test sırasında modelin performansını izlemek için kullanılır (accuracy)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 4. Modeli Eğit (Train)
# x_train: Eğitim verisi
# y_train: Eğitim etiketleri
# epochs: Modelin tüm veri kümesi üzerinde kaç kez döneceğini belirler
# batch_size: Her eğitim adımında kaç örnek kullanılacağını belirler
# validation_split: Eğitim verisinin bir kısmını doğrulama için ayırır (eğitim sırasında performansı izlemek için)
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 5. Modeli Değerlendir (Evaluate)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest doğruluğu: {test_acc:.4f}")

# 6. Tahmin Yap ve Görselleştir (İsteğe Bağlı)
predictions = model.predict(x_test)

# Örnek bir tahmin ve görselleştirme
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Tahmin: {np.argmax(predictions[i])}\nGerçek: {y_test[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# Eğitim geçmişini görselleştir (kayıp ve doğruluk grafikleri)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.title('Kayıp Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.title('Doğruluk Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')

plt.tight_layout()
plt.show()
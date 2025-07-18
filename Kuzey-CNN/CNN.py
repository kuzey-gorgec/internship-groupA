import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# Veri Yükleme (MNIST veri seti: 28x28 gri tonlamalı rakam görüntüleri)
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# CNN Modeli Oluşturma
model = models.Sequential()

# Convolutional Layer

# Bu katman, giriş görüntüsünden kenar, köşe gibi temel desenleri çıkarır.
# 3x3 filtreler ile görüntü üzerinde kayan pencere işlemi yapılır.
# Her filtre: (3x3x1 + 1) = 10 parametre, toplam 32 filtre ile 320 parametre öğrenilir.
# Aktivasyon olarak ReLU kullanılarak doğrusal olmayanlık kazandırılır.
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# MaxPooling Layer

# 2x2 boyutlu bölgelerden maksimum değeri alır.
# Bu işlem görüntü boyutunu azaltır (hesaplamayı düşürür), özellikleri özetler.
# Öğrenilecek parametre yoktur.
model.add(layers.MaxPooling2D((2, 2)))

# İkinci Convolution + Pooling

# Daha derin özellikleri öğrenmek için ikinci kez evrim uygulanır.
# 64 filtre ile daha fazla desen tanımlanabilir hale gelir.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten Layer 

# 2 boyutlu feature map -> 1 boyutlu vektör dönüşümü
# Örn: 7x7x64 -> 3136 boyutlu vektör
model.add(layers.Flatten())

# Fully Connected (Dense) Layer

# Gizli katman: Giriş vektörü tüm nöronlara bağlanır.
# Parametre sayısı: (3136 + 1) x 64 = 200768
# Aktivasyon olarak ReLU ile doğrusal olmayanlık korunur.
model.add(layers.Dense(64, activation='relu'))

# Çıkış Katmanı

# 10 sınıf için (0-9 rakamları) softmax fonksiyonu kullanılır.
# Softmax: her sınıfa ait olasılık hesaplar ve toplamı 1 olur.
model.add(layers.Dense(10, activation='softmax'))

# Modelin derlenmesi
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model Eğitimi (Epoch=5)
# Epoch sayısı: modelin veri üzerinde kaç kez eğitileceğini belirler.
# Düşük epoch sayısı -> az öğrenme (underfitting)
# Yüksek epoch sayısı -> aşırı öğrenme riski (overfitting)
# Biz 5 epoch ile eğitim yapıyoruz.
history_5 = model.fit(x_train, y_train, epochs=5,
                      validation_data=(x_test, y_test))

#Modeli kaydetme
model.save('mnist_cnn_model.h5')
print("Model 'mnist_cnn_model.h5' olarak kaydedildi.")

# Doğruluk Grafiği (5 epoch için)
plt.plot(history_5.history['accuracy'], label='Eğitim (5 epoch)')
plt.plot(history_5.history['val_accuracy'], label='Test (5 epoch)')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.title("CNN Doğruluk Karşılaştırması (5 Epoch)")
plt.show()

# Rastgele 10 test örneği alalım
num_examples = 10
random_indices = np.random.choice(len(x_test), num_examples, replace=False)
random_images = x_test[random_indices]
random_labels = y_test[random_indices]

# Tahmin yap
predictions = model.predict(random_images)

# Tahmin Sonuçlarını Görselleştirme
plt.figure(figsize=(15, 5))
for i in range(num_examples):
    plt.subplot(2, 5, i+1)
    plt.imshow(random_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Gerçek: {random_labels[i]}\nTahmin: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.suptitle("Model İle Tahminler")
plt.show()

# Epoch sayısı arttıkça eğitim doğruluğu genellikle artar
# Ancak test doğruluğu bir noktadan sonra sabit kalabilir ya da düşebilir (overfitting)

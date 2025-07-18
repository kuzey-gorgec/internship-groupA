CNN KATMANLARI

Convolutional Layer

# Bu katman, giriş görüntüsünden kenar, köşe gibi temel desenleri çıkarır.
# 3x3 filtreler ile görüntü üzerinde kayan pencere işlemi yapılır.
# Her filtre: (3x3x1 + 1) = 10 parametre, toplam 32 filtre ile 320 parametre öğrenilir.
# Aktivasyon olarak ReLU kullanılarak doğrusal olmayanlık kazandırılır.

MaxPooling Layer

# 2x2 boyutlu bölgelerden maksimum değeri alır.
# Bu işlem görüntü boyutunu azaltır (hesaplamayı düşürür), özellikleri özetler.
# Öğrenilecek parametre yoktur.

İkinci Convolution + Pooling

# Daha derin özellikleri öğrenmek için ikinci kez evrim uygulanır.
# 64 filtre ile daha fazla desen tanımlanabilir hale gelir.

Flatten Layer 

# 2 boyutlu feature map -> 1 boyutlu vektör dönüşümü
# Örn: 7x7x64 -> 3136 boyutlu vektör

Fully Connected (Dense) Layer

# Gizli katman: Giriş vektörü tüm nöronlara bağlanır.
# Parametre sayısı: (3136 + 1) x 64 = 200768
# Aktivasyon olarak ReLU ile doğrusal olmayanlık korunur.

Çıkış Katmanı

# 10 sınıf için (0-9 rakamları) softmax fonksiyonu kullanılır.
# Softmax: her sınıfa ait olasılık hesaplar ve toplamı 1 olur.

Modelin derlenmesi
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

kodu ile yapılıyor.

# Model Eğitimi (Epoch=5)
# Epoch sayısı: modelin veri üzerinde kaç kez eğitileceğini belirler.
# Düşük epoch sayısı -> az öğrenme (underfitting)
# Yüksek epoch sayısı -> aşırı öğrenme riski (overfitting)
# Biz 5 epoch ile eğitim yapıyoruz.
# Not:
# Epoch sayısı arttıkça eğitim doğruluğu genellikle artar
# Ancak test doğruluğu bir noktadan sonra sabit kalabilir ya da düşebilir (overfitting)

Detection - Nesne Tespiti Nedir?
- Görüntüdeki nesnelerin konumlarını (bounding box) ve sınıflarını belirler.
- Classification'dan farkı, nesnenin hangi bölgede olduğunu da bulur.
- Yüz tespiti, araç tespiti gibi uygulamalarda kullanılır.

Haar Cascade Algoritması:
- Görüntüyü küçük pencerelere böler.
- Her pencerede Haar özelliklerini hesaplar (örneğin, kenar, çizgi farkları).
- AdaBoost ile eğitilmiş güçlü zayıf sınıflandırıcılar kullanır.
- scaleFactor: Pencere ölçeğinin her adımda ne kadar küçüleceği.
- minNeighbors: Pozitif pencerelerin sayısı (gürültüyü azaltır).

Segmentation - Segmentasyon Nedir?
- Görüntüdeki her pikselin hangi sınıfa ait olduğunu belirler.
- Classification ve Detection’dan farkı: tüm pikseller etiketlenir.
- Semantic segmentation: Piksel bazında sınıf tahmini (örn. yol, araba, insan).
- Instance segmentation: Aynı sınıftan nesneleri ayrı ayrı ayırır.

DeepLabV3 Modeli:
- Atrous (dilated) convolution kullanarak daha geniş alanlardan bilgi toplar.
- Atrous Spatial Pyramid Pooling (ASPP) ile çok ölçekli özellikler yakalar.
- Önceden ImageNet ve COCO üzerinde eğitilmiş modeller hazır olarak kullanılır.

Model PyTorch torchvision paketinden yüklenilir.


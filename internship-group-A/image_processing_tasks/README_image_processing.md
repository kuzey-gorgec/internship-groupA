# Basit CNN ile Temel Görüntü İşleme Görevleri (Sınıflandırma, Tespit, Segmentasyon)

Bu proje, Evrişimsel Sinir Ağlarının (CNN) temel yapısını ve bilgisayar görüşündeki en yaygın görevlerden olan Görüntü Sınıflandırma, Nesne Tespiti ve Görüntü Segmentasyonu'nu pratik kod örnekleri aracılığıyla açıklamayı amaçlamaktadır.

## Proje Amacı

- **Evrişim (Convolution), Havuzlama (Pooling), Aktivasyon (Activation) ve Tam Bağlantılı (Fully Connected) katmanlarının** CNN'ler içindeki rollerini ve pratik uygulamalarını kavramak.
- **Görüntü Sınıflandırma, Nesne Tespiti ve Görüntü Segmentasyonu** görevleri arasındaki temel farkları, çıktı türlerini ve uygulama senaryolarını kod örnekleriyle gözlemlemek.
- Derin öğrenme modellerinin oluşturulması, derlenmesi, eğitilmesi ve değerlendirilmesi gibi temel adımları deneyimlemek.

## Kullanılan Teknolojiler

- **Python 3.x**
- **TensorFlow / Keras:** Derin öğrenme modellerini oluşturmak ve eğitmek için ana kütüphane.
- **NumPy:** Sayısal işlemler ve veri manipülasyonu için.
- **Matplotlib:** Sonuçları ve eğitim süreçlerini görselleştirmek için.
- **OpenCV (`cv2`):** Görüntü yükleme ve temel görsel işlemleri için (özellikle segmentasyon örneğinde dummy görsel oluşturmak için).

## Kurulum

Projeyi yerel ortamınızda çalıştırabilmek için aşağıdaki kütüphaneleri kurmanız gerekmektedir. Terminalinizi veya komut istemcinizi açın ve aşağıdaki komutları çalıştırın:

```bash
pip install tensorflow keras numpy matplotlib opencv-python

--

## Kodların Açıklaması:

Bu proje, her biri farklı bir görüntü işleme görevine odaklanan üç ayrı Python betiği içermektedir.

1. image_classification_example.py (Görüntü Sınıflandırma)
Amaç: Bir görüntünün (bu örnekte MNIST el yazısı rakamları) genel olarak hangi kategoriye ait olduğunu tahmin etmek. Çıktı tek bir sınıf etiketidir.

Kavramlar:

Veri Yükleme ve Hazırlık: MNIST veri kümesinin yüklenmesi ve CNN için uygun formata (28x28x1) getirilerek normalizasyonu.

layers.Conv2D (Evrişim Katmanı): Görüntüdeki kenarlar ve dokular gibi yerel özellikleri algılamak için filtrelerin kullanılması. ReLU aktivasyonu ile doğrusal olmayan özellikler eklenmesi.

layers.MaxPooling2D (Havuzlama Katmanı): Özellik haritalarının boyutunu küçülterek hesaplama yükünü azaltma ve konum invaryantlığı sağlama.

layers.Flatten(): 2D özellik haritalarını Tam Bağlantılı katmanlara beslemek için 1D vektöre dönüştürme.

layers.Dense (Tam Bağlantılı Katmanlar): Yüksek seviyeli öğrenilen özelliklerden yola çıkarak nihai sınıflandırma kararını verme. Çıkış katmanında softmax aktivasyonu ile sınıf olasılıkları üretme.

Model Derleme ve Eğitim: adam optimizer ve sparse_categorical_crossentropy loss fonksiyonu ile modelin derlenmesi ve model.fit() ile eğitilmesi.

Değerlendirme ve Tahmin: Test verisi üzerinde modelin doğruluğunun ölçülmesi ve örnek tahminlerin görselleştirilmesi.

2. object_detection_example.py (Nesne Tespiti)
Amaç: Bir görüntüdeki nesnelerin nerede olduğunu (sınırlayıcı kutu ile) ve ne olduğunu (sınıf etiketi) belirlemek.

Kavramlar:

Önceden Eğitilmiş Model Kullanımı: Keras'ın applications modülünden ImageNet üzerinde eğitilmiş MobileNetV2 gibi bir modelin yüklenmesi. Bu tür modeller, geniş bir nesne yelpazesini tanıyabilir.

Görüntü Ön İşleme: Modelin beklediği boyuta (224x224) ve formata göre görüntünün hazırlanması.

Tahmin (model.predict): Modelin girdi görüntüdeki nesnelerin olasılıklarını tahmin etmesi.

Sonuçların Dekode Edilmesi: Modelin sayısal çıktılarını okunabilir etiketlere ve güven skorlarına dönüştürme (decode_predictions).

Görselleştirme: Tahmin edilen nesnelerin isimleri ve skorlarının görüntü üzerine yazdırılması.

Not: Bu örnek, gerçek bir nesne tespiti modelinin (örneğin YOLO, SSD) doğrudan bounding box koordinatları çıktısını sağlamaz, ancak bir görüntüdeki baskın nesnelerin sınıflandırılması yoluyla "tespit" kavramını gösterir.

3. image_segmentation_example.py (Görüntü Segmentasyonu)
Amaç: Bir görüntüdeki her pikseli belirli bir sınıfa atayarak nesnelerin ve bölgelerin tam şekillerini ve sınırlarını çıkarmak.

Kavramlar:

TensorFlow Hub: Önceden eğitilmiş segmentasyon modellerine kolayca erişim sağlayan bir platform. Bu örnekte DeeplabV3 gibi bir model kullanılır.

Piksel Düzeyinde Sınıflandırma: Görüntüdeki her pikselin ait olduğu kategoriye göre etiketlenmesi.

Maske Oluşturma: Modelin çıktısı olarak orijinal görüntü boyutunda bir "segmentasyon maskesi" veya "harita" elde etme. Her pikselin değeri bir sınıfı temsil eder.

Görüntü Ön İşleme ve Yeniden Boyutlandırma: Modelin beklediği giriş boyutuna göre görüntünün yeniden boyutlandırılması.

Görselleştirme: Segmentasyon maskesinin farklı renklerle temsil edilmesi ve orijinal görüntü üzerine şeffaf bir şekilde bindirilerek sonuçların gösterilmesi.
```

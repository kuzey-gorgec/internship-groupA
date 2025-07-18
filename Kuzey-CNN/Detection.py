import cv2
import matplotlib.pyplot as plt

"""
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
"""

# Haar cascade xml dosyasını yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Örnek resim yükleme
img = cv2.imread('person.jpg')
if img is None:
    print("Dosya bulunamadı! Lütfen tekrar kontrol ediniz.")
    exit()

# Gri tonlamaya çevirme: Haar Cascade gri görüntüde çalışır
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Yüzleri tespit etme
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,   # Pencere boyutunu %10 küçülterek tarar
    minNeighbors=10    # En az olması gereken komşuların sayısını belirleme yapılır.Komşu sayısına göre tespit edilen yüz sayısı değişebilir.
)

print(f"Tespit edilen yüz sayısı: {len(faces)}")

# Tespit edilen yüzleri dikdörtgen içine al
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)  # Kırmızı kutu

# Görüntüyü RGB formatına çevir (Matplotlib BGR desteklemez)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Haar Cascade ile Yüz Tespiti")
plt.axis('off')
plt.show()

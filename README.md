# internship-groupA
# SIFT, ORB ve CNN Açıklamaları

## SIFT ve ORB Nedir?

SIFT ve ORB algoritmaları, bir görüntüdeki **benzersiz ve önemli noktaları (keypoints)** bulup başka görüntülerle eşleştirir.

### SIFT (Scale-Invariant Feature Transform)
- Ölçek ve dönmeye karşı dayanıklı noktalar bulur.
- Özellikle köşeler ve yüksek kontrastlı bölgelerde çalışır.
- Her nokta için 128 boyutlu özellik çıkarır.
- Eşleşme için Öklid mesafesi (L2 distance) kullanılır.

### ORB (Oriented FAST and Rotated BRIEF)
- SIFT'e göre daha hızlı ve hafiftir.
- FAST algoritmasıyla köşe tespit eder.
- BRIEF ile ikili (binary) özellik çıkarır.
- Eşleşme için Hamming mesafesi kullanılır.
- Daha fazla nokta bulma eğilimindedir.

---

## CNN (Convolutional Neural Networks) Nedir?

CNN, özellikle görsel veriler için geliştirilmiş bir yapay sinir ağı türüdür. Görüntüdeki yapısal özellikleri öğrenerek nesne tanıma, sınıflandırma yapar.

---

## CNN Katmanları ve Matematiksel Anlatımı

### 1. Convolution (Evrişim) Katmanı
- Görüntü üzerinde küçük filtreler kaydırılarak özellik haritaları çıkarılır.
- Matematiksel ifade:

S(i,j) = Σ_m Σ_n I(i+m, j+n) * K(m, n)

- I: Giriş görüntüsü
- K: Filtre (kernel)
- S: Özellik haritası (feature map)

### 2. Pooling (Alt Örnekleme) Katmanı
- Özellik haritalarını küçültür, hesaplamayı azaltır.
- En yaygın: Max Pooling (2x2 bölgeden maksimum seçilir).

P(i,j) = max { S(2i, 2j), S(2i, 2j+1), S(2i+1, 2j), S(2i+1, 2j+1) }


### 3. Activation Function (Aktivasyon Fonksiyonu)
- Ağda doğrusal olmayanlık sağlar.
- En çok kullanılan: ReLU

f(x) = max(0, x)


### 4. Fully Connected (Tam Bağlantılı) Katman
- Tüm nöronlar birbirine bağlıdır.
- Sınıflandırma yapılır.

y = f(Wx + b)


- x: Giriş vektörü
- W: Ağırlık matrisi
- b: Bias
- f: Aktivasyon fonksiyonu
- y: Çıkış

---

## CNN'in Çalışma Prensibi

1. **Özellik Çıkarma:**
 - Convolution katmanları ile görüntüdeki kenar, doku gibi özellikler çıkarılır.
 - Aktivasyon fonksiyonları doğrusal olmayanlık sağlar.
 - Pooling ile boyut küçültülür, önemli bilgiler korunur.

2. **Sınıflandırma:**
 - Flatten katmanı ile veriler vektöre dönüştürülür.
 - Fully connected katmanlarla sınıflandırma yapılır.
 - Softmax ile her sınıfa ait olasılık hesaplanır.

---

**Not:** CNN genellikle gri tonlama (grayscale) görüntüleri kullanır, renk bilgisi doğrudan kullanılmaz.







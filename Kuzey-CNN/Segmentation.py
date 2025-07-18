import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

"""
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
"""

# Önceden eğitilmiş modeli alıyoruz.
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

# Örnek resim dosyasını çalışma klasörüne koyuyoruz.
try:
    image = Image.open('person.jpg').convert('RGB')
except FileNotFoundError:
    print("Dosya bulunamadı! Lütfen tekrar kontrol ediniz.")
    exit()

# Görüntüyü modele uygun şekilde hazırlama
preprocess = transforms.Compose([
    transforms.Resize(256),       # Boyutlandırma
    transforms.ToTensor(),        # Tensor'a çevirme
    transforms.Normalize(         # Normalize etme (ImageNet ortalaması ve std)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = preprocess(image).unsqueeze(0)  # Modelin beklediği 4D tensor [1,3,H,W]

# Tahmin yapma adımı
with torch.no_grad():
    output = model(input_tensor)['out'][0]  # [Channel, Height, Width] kanal sayısı = sınıf sayısı

# Her piksel için en olası sınıfı seçme
segmentation_map = output.argmax(0).byte().cpu().numpy()

# Görselleştirme
plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Orijinal Görüntü")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmentation_map)
plt.title("Semantic Segmentasyon Haritası (Sınıf ID’leri)")
plt.axis('off')

plt.show()

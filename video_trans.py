import cv2
import os
import glob

def frames_to_video(image_folder, output_video_path, fps=30):
    """
    Parça parça resimlerden oluşan klasörü okuyarak MP4 formatında video üretir.
    """
    # Klasördeki görselleri uzantılarına göre bul ve alfabetik/sayısal sıraya diz
    # (Karelerin sırasının bozulmaması için sorted() kullanımı kritiktir)
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(image_folder, ext)))
    images = sorted(images)

    # Klasörde görsel yoksa işlemi iptal et
    if not images:
        print(f"Hata: '{image_folder}' klasöründe geçerli resim bulunamadı!")
        return

    # İlk resmi okuyarak videonun genişlik ve yükseklik boyutlarını otomatik al
    first_image = cv2.imread(images[0])
    height, width, layers = first_image.shape
    size = (width, height)

    # Video yazıcıyı (VideoWriter) tanımla. MP4 formatı için 'mp4v' kodeği kullanılır.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    print(f"Video dönüştürme başladı. Toplam kare sayısı: {len(images)}")
    
    # Tüm resimleri sırayla videoya ekle
    for img_path in images:
        frame = cv2.imread(img_path)
        video.write(frame)

    # İşlem bittiğinde belleği temizle ve dosyayı kapat
    video.release()
    print(f"Başarılı! Video kaydedildi: {output_video_path}")

# ---- KULLANIM ÖRNEĞİ ----
# Kendi bilgisayarınızdaki klasör yollarına göre buraları düzenleyebilirsiniz:
GÖRSEL_KLASÖRÜ = "C:\\Users\\HP\\Downloads\\VisDrone2019-VID-val\\VisDrone2019-VID-val\\sequences\\uav0000305_00000_v"
ÇIKTI_VİDEO_YOLU = "visdrone_output_video2.mp4"
VİDEO_FPS_DEĞERİ = 30 # Saniyede akacak kare sayısı (VisDrone videoları genelde 25-30 FPS'tir)

frames_to_video(GÖRSEL_KLASÖRÜ, ÇIKTI_VİDEO_YOLU, fps=VİDEO_FPS_DEĞERİ)

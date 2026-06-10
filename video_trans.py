import cv2
import os
import glob

def frames_to_video(image_folder, output_video_path, fps=30):

    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(image_folder, ext)))
    images = sorted(images)

    if not images:
        print(f"Hata: '{image_folder}' klasöründe geçerli resim bulunamadı!")
        return

    first_image = cv2.imread(images[0])
    height, width, layers = first_image.shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    print(f"Video dönüştürme başladı. Toplam kare sayısı: {len(images)}")

    for img_path in images:
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Başarılı! Video kaydedildi: {output_video_path}")


GÖRSEL_KLASÖRÜ = "C:\\Users\\HP\\Downloads\\VisDrone2019-VID-val\\VisDrone2019-VID-val\\sequences\\uav0000305_00000_v"
ÇIKTI_VİDEO_YOLU = "visdrone_output_video2.mp4"
VİDEO_FPS_DEĞERİ = 30 

frames_to_video(GÖRSEL_KLASÖRÜ, ÇIKTI_VİDEO_YOLU, fps=VİDEO_FPS_DEĞERİ)

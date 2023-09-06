import cv2
import os
import sys

# berikan jalur gambar sebagai argumen
image_path = sys.argv[1]

# membuat gambar output dengan nama yang sama dengan gambar input
output_image_name = os.path.splitext(image_path)[0] + "_detected.jpg"

# memuat gambar yang akan diuji
image = cv2.imread(image_path)

# mengubah ke skala keabuan
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# inisialisasi pengenal wajah (cascade wajah default)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")

# mendeteksi semua wajah pada gambar (variabel image)
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

# mencetak jumlah wajah yang terdeteksi
print(f"{len(faces)} faces detected in the image.")

# untuk setiap wajah akan menggambar persegi
for x, y, width, height in faces:
    cv2.rectangle(
        image, (x, y), (x + width, y + height), color=(0, 128, 0), thickness=4
    )

# menyimpan gambar dengan persegi panjang
cv2.imwrite(output_image_name, image)

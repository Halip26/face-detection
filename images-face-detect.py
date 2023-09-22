import cv2
import os
import sys

# berikan jalur gambar sebagai argumen
image_path = sys.argv[1]

output_directory = "output/"

os.makedirs(output_directory, exist_ok=True)

# Ekstrak nama file dari image_path
filename = os.path.basename(image_path)

# Memisahkan nama file dan ekstensi
name, extension = os.path.splitext(filename)

# Gabungkan direktori output & nama file yg ditambah akhiran "_detected"
output_image_path = os.path.join(output_directory, f"{name}_detected{extension}")

# memuat gambar yang akan diuji
image = cv2.imread(image_path)

# mengubah ke skala keabuan
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# inisialisasi pengenal wajah (cascade wajah default)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

# mendeteksi semua wajah pada gambar (variabel image)
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

# mencetak jumlah wajah yang terdeteksi
if len(faces) > 1:
    print(f"{len(faces)} faces detected on the camera")
else:
    print(f"{len(faces)} face detected on the camera")

# untuk setiap wajah akan menggambar persegi
for x, y, width, height in faces:
    cv2.rectangle(
        image, (x, y), (x + width, y + height), color=(0, 128, 0), thickness=4
    )

# menyimpan gambar dengan persegi panjang
cv2.imwrite(output_image_path, image)

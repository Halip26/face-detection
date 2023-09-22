import cv2
import numpy as np
import os
import sys

# Jalur prototxt model Caffe
prototxt_path = "weights/deploy.prototxt.txt"

# Jalur model Caffe
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# memuat model caffe
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# berikan jalur gambar sebagai argumen
image_path = sys.argv[1]

output_directory = "output/"

os.makedirs(output_directory, exist_ok=True)

# Ekstrak nama file dari image_path
filename = os.path.basename(image_path)

# Memisahkan nama file dan ekstensi
name, extension = os.path.splitext(filename)

# Gabungkan direktori output & nama file yg ditambah akhiran "_detected"
output_image_path = os.path.join(output_directory, f"{name}_dnn_detected{extension}")

# memuat gambar yang akan diuji
image = cv2.imread(image_path)

# mendapatkan lebar & tinggi gambar
height, width, _ = image.shape

# mengubah ke skala keabuan
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# inisialisasi pengenal wajah (cascade wajah default)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

# mendeteksi semua wajah pada gambar (variabel image)
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

# pra-pemrosesan gambar: resize dan pengurangan mean (rata-rata)
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# menetapkan gambar menjadi input jaringan saraf
model.setInput(blob)

# melakukan inferensi & mendaptkan hasilnya
output = np.squeeze(model.forward())

# mengatur ukuran font & font style
font_scale = 1
font_style = cv2.FONT_HERSHEY_SIMPLEX

# buat persegi panjang untuk mendeteksi wajah dengan perulangan
for i in range(0, output.shape[0]):
    # membuat var kepercayaan untuk output looping i
    face_accuracy = output[i, 2]

    # jika kepercayaan di atas 50%, maka gambarkan kotak sekitarnya
    if face_accuracy > 0.5:
        # get koordinat kotak sekitarnya dan memperbesar ukurannya ke gambar asli
        box = output[i, 3:7] * np.array([width, height, width, height])
        # mengkonversi ke integer
        start_x, start_y, end_x, end_y = box.astype(np.int64)
        # menggambar persegi panjang disekitar wajah
        cv2.rectangle(
            image, (start_x, start_y), (end_x, end_y), color=(0, 128, 0), thickness=4
        )
        # membuat teksnya juga diatas persegi panjang
        cv2.putText(
            image,
            f"Face {face_accuracy*100:.2f}%",
            (start_x, start_y - 5),
            font_style,
            font_scale,
            (0, 128, 0),
            2,
        )
if len(faces) > 1:
    print(f"{len(faces)} faces detected on the camera")
else:
    print(f"{len(faces)} face detected on the camera")

# mengatur ukuran jendela sesuai dengan gambar asli
cv2.namedWindow("The results", cv2.WINDOW_NORMAL)
cv2.resizeWindow("The results", width, height)

# # menampilkan gambarnya pada jendela baru
cv2.imshow("The results", image)
cv2.waitKey(0)

# menyimpan gambar beserta persegi panjangnya
cv2.imwrite(output_image_path, image)

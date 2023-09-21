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
output_image_path = os.path.join(output_directory, f"{name}_detected{extension}")

# memuat gambar yang akan diuji
image = cv2.imread(image_path)
# mendapatkan lebar & tinggi gambar
h, w = image.shape[:2]

# pra-pemrosesan gambar: resize dan pengurangan mean (rata-rata)
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# menetapkan gambar menjadi input jaringan saraf
model.setInput(blob)

# melakukan inferensi & mendaptkan hasilnya
output = np.squeeze(model.forward())

# mendefinisikan variabel font_scale
font_scale = 1

# buat persegi panjang untuk mendeteksi wajah dengan perulangan
for i in range(0, output.shape[0]):
    # membuat var kepercayaan untuk output looping i
    face_accuracy = output[i, 2]

    # jika kepercayaan di atas 50%, maka gambarkan kotak sekitarnya
    if face_accuracy > 0.5:
        # get koordinat kotak sekitarnya dan memperbesar ukurannya ke gambar asli
        box = output[i, 3:7] * np.array([w, h, w, h])
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
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 128, 0),
            2,
        )

# menampilkan gambarnya pada jendela baru
cv2.imshow("The results", image)
cv2.waitKey(0)

# menyimpan gambar beserta persegi panjangnya
cv2.imwrite(output_image_path, image)

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

# membuat gambar output dengan nama yang sama dengan gambar input
output_image_name = os.path.splitext(image_path)[0] + "_detected.jpg"

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
font_scale = 0.8

# buat persegi panjang untuk mendeteksi wajah dengan perulangan
for i in range(0, output.shape[0]):
    # membuat var kepercayaan untuk output looping i
    confidence = output[i, 2]

    # jika kepercayaan di atas 50%, maka gambarkan kotak sekitarnya
    if confidence > 0.5:
        # get koordinat kotak sekitarnya dan memperbesar ukurannya ke gambar asli
        box = output[i, 3:7] * np.array([w, h, w, h])
        # mengkonversi ke integer
        start_x, start_y, end_x, end_y = box.astype(np.int)
        # menggambar persegi panjang disekitar wajah
        cv2.rectangle(
            image, (start_x, start_y), (end_x, end_y), color=(0, 128, 0), thickness=4
        )
        # membuat teksnya juga diatas persegi panjang
        cv2.putText(
            image,
            f"{confidence*100:.2f}%",
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
cv2.imwrite(output_image_name, image)

import cv2
import numpy as np

# Jalur prototxt model Caffe
prototxt_path = "weights/deploy.prototxt.txt"

# Jalur model Caffe
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# menginisialisasi pengenalan wajah (defult face haarcascade)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

# memuat model caffe
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# membuat objek kamera baru
capture = cv2.VideoCapture(0)

while True:
    # membaca gambar/frame dari kamera
    live, image = capture.read()
    # mendapatkan lebar & tinggi gambar
    h, w = image.shape[:2]

    # menkonversi gambar ke skala abu2
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # mendeteksi semua wajah pada kamera
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    # pra-pemrosesan gambar: resize dan pengurangan mean (rata-rata)
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # menetapkan gambar menjadi input jaringan saraf
    model.setInput(blob)

    # melakukan inferensi & mendaptkan hasilnya
    output = np.squeeze(model.forward())

    # mendefinisikan variabel font_scale
    font_scale = 0.6

    # buat persegi panjang untuk mendeteksi wajah dengan perulangan
    for i in range(0, output.shape[0]):
        # membuat var facialAccuracy untuk output looping i
        facialAccuracy = output[i, 2]

        # jika akurasi wajah di atas 50%, maka gambarkan kotak sekitarnya
        if facialAccuracy > 0.5:
            # get koordinat kotak sekitarnya dan memperbesar ukurannya ke gambar asli
            box = output[i, 3:7] * np.array([w, h, w, h])
            # mengkonversi ke integer
            start_x, start_y, end_x, end_y = box.astype(np.int64)
            # menggambar persegi panjang disekitar wajah
            cv2.rectangle(
                image,
                (start_x, start_y),
                (end_x, end_y),
                color=(0, 128, 0),
                thickness=4,
            )
            # membuat teksnya juga diatas persegi panjang
            cv2.putText(
                image,
                f"Tamvan? {facialAccuracy*100:.2f}%",
                (start_x, start_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 128, 0),
                2,
            )

            # mencetak jumlah wajah yang terdeteksi
            if len(faces) > 1:
                print(f"{len(faces)} faces detected on the camera.")
            else:
                print(f"{len(faces)} face detected on the camera.")

            # mencetak jumlah wajah yang terdeteksi beserta tingkat akurasinya
            """if len(faces) > 1:
                print(
                    f"{len(faces)} faces detected on the camera, \n{facialAccuracy*100:.2f}% facialAccuracy"
                )
            else:
                print(
                    f"{len(faces)} face detected on the camera, \n{facialAccuracy*100:.2f}% facialAccuracy"
                )"""

    # menampilkan jendela baru
    cv2.imshow("Face detect v1.0", image)

    # jika pegguna menekan tombol "q" maka perulangan akan berhenti
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

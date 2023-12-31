import cv2

# menginisialisasi pengenalan wajah (defult face haarcascade)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

# membuat objek kamera baru
capture = cv2.VideoCapture(0)

while True:
    # membaca gambar/frame dari kamera
    live, camera = capture.read()
    # menkonversi gambar ke skala abu2
    image_gray = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)

    # mendeteksi semua wajah pada kamera
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    # menghitung jumlah wajah yg terdeteksi
    face_count = len(faces)

    # untuk setiap wajah, menggambar persegi panjang
    for x, y, width, height in faces:
        cv2.rectangle(
            camera, (x, y), (x + width, y + height), color=(0, 128, 0), thickness=3
        )
    # menampilkan jendela baru
    cv2.imshow("Face detect v1.0", camera)

    # mencetak jumlah wajah yang terdeteksi
    if face_count > 1:
        print(f"{face_count} faces detected on the camera", end="\r")
    else:
        print(f"{face_count} face detected on the camera", end="\r")

    # jika pegguna menekan tombol "q", maka perulangan akan berhenti
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

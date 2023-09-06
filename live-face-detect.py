import cv2

# membuat objek kamera baru
capture = cv2.VideoCapture(0)

# menginisialisasi pengenalan wajah (defult face haarcascade)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")

while True:
    # membaca gambar/frame dari kamera
    _, image = capture.read()
    # menkonversi gambar ke skala abu2
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # mendeteksi semua wajah pada kamera
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    # untuk setiap wajah, menggambar persegi panjang
    for x, y, width, height in faces:
        cv2.rectangle(
            image, (x, y), (x + width, y + height), color=(0, 128, 0), thickness=3
        )
    # menampilkan jendela baru
    cv2.imshow("Pendeteksi Wajah", image)

    # jika pegguna menekan tombol q
    # maka perulangan akan berhenti
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

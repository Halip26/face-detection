import cv2

# memuat gambar yg akan diuji
image = cv2.imread("me.jpg")

# mengubah ke skala keabuan
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# inisiliasasi pengenal wajah (cascade wajah default)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")

# mendeteksi semua wajah pada gambar (variabel image)
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

# mencetak jumlah wajah yg terdeteksi
print(f"{len(faces)} faces detected in the image.")

# untuk setiap wajah akan menggambar persegi
for x, y, width, height in faces:
    cv2.rectangle(
        image, (x, y), (x + width, y + height), color=(0, 128, 0), thickness=6
    )

# menyimpan gambar dengan persegi panjang
cv2.imwrite("me_detected.jpg", image)

# 1. IMPORT LIBRARY
import cv2 # Manggil OpenCV buat urusan kamera
from ultralytics import YOLO # Manggil algoritma YOLO

def main():
    # 2. LOAD MODEL AI
    # YOLO('yolov8n.pt') nyuruh program nge-download model 'nano' (paling kecil & cepat).
    # Model ini udah diajarin ngenalin 80 objek umum (orang, HP, kursi, dll) sama pembuatnya.
    print("Sedang memuat model YOLO...")
    model = YOLO('yolov8n.pt') 

    # 3. NYALAKAN KAMERA
    # cv2.VideoCapture(0) itu manggil hardware kamera.
    # Angka 0 = Kamera bawaan laptop. Kalau lo pake webcam USB external, biasanya angkanya 1 atau 2.
    cap = cv2.VideoCapture(0)

    print("Kamera aktif. Tekan 'q' untuk berhenti.")

    # 4. LOOPING (Video itu pada dasarnya adalah foto yang dijepret terus-menerus)
    while cap.isOpened():
        # cap.read() ngambil 1 jepretan foto saat itu juga (frame)
        status_sukses, frame = cap.read() 
        
        # Kalau kamera mati/error, loopnya dihentikan
        if not status_sukses:
            print("Waduh, kamera nggak kebaca nih!")
            break

        # 5. PROSES DETEKSI
        # AI disuruh nebak isi 'frame' (foto tadi).
        # conf=0.5 artinya: AI cuma bakal ngasih tau kalau dia YAKIN 50% ke atas.
        # verbose=False biar terminal lo nggak kepenuhan teks proses deteksi.
        hasil_prediksi = model.predict(source=frame, conf=0.5, verbose=False)

        # 6. GAMBAR KOTAK
        # Hasil deteksi AI (koordinat x,y kotak) digambar langsung ke foto aslinya.
        frame_ada_kotaknya = hasil_prediksi[0].plot()

        # 7. TAMPILKAN KE LAYAR
        # Bikin jendela aplikasi bernama "Tes Setup PI" yang isinya foto yang udah ada kotaknya.
        cv2.imshow("Tes Setup PI", frame_ada_kotaknya)

        # 8. TOMBOL EXIT
        # Sistem nunggu 1 milidetik. Kalau dalam 1 milidetik itu lo nekan tombol 'q' di keyboard, program berhenti.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. BERSIH-BERSIH (Best Practice)
    # Jangan biarin kamera nyala terus pas program mati, nanti RAM bocor.
    cap.release()
    cv2.destroyAllWindows()

# Ini best practice di Python. Artinya: jalankan fungsi main() HANYA JIKA file ini dijalankan langsung.
if __name__ == "__main__":
    main()
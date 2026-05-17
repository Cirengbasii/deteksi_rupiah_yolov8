# ==========================================
# 1. IMPORT AMUNISI (LIBRARY)
# ==========================================
import cv2          # Buat nyalain webcam dan nampilin kotak di layar
import time         # Buat ngatur jeda waktu (cooldown) suara
import threading    # JALAN NINJA: Buat ngomong di background biar kamera gak patah-patah
import pyttsx3      # Buat ngubah teks jadi suara robot (Text-to-Speech)
from ultralytics import YOLO # Algoritmanya buat nyawa deteksi

# ==========================================
# 2. SETUP ENGINE SUARA (TEXT TO SPEECH)
# ==========================================
# pyttsx3.init() itu buat ngidupin mesin suara di laptop lo
mesin_suara = pyttsx3.init()
# SetProperty 'rate' buat ngatur kecepatan ngomong. 150 kata per menit itu udah pas banget.
mesin_suara.setProperty('rate', 150) 

# Variabel bantuan buat ngatur jeda suara (Best Practice)
waktu_terakhir_ngomong = 0
jeda_suara = 3.0 # AI baru boleh ngomong lagi setelah 3 detik dari omongan terakhir

# Fungsi khusus buat ngomong. Harus dipisah biar bisa dijalankan lewat Threading
def fungsi_ngomong(teks):
    mesin_suara.say(teks) # Nyiapin teks yang mau diomongin
    mesin_suara.runAndWait() # Perintah buat ngomong sampai selesai

# ==========================================
# 3. CORE PROGRAM (FUNGSI UTAMA)
# ==========================================
def main():
    global waktu_terakhir_ngomong
    
    # Kita paksa panggil file best.pt yang barusan lo paste di folder utama
    print("Memuat model AI hasil training...")
    model = YOLO(r"D:\PROJECT SENDIRI\BISMILLAHIRAHMANNIRRAHIM PI ARIF MUHAMMAD AKBAR\PI_Arif\best.pt")

    # Nyalain webcam laptop
    cap = cv2.VideoCapture(0)
    print("Kamera aktif! Arahkan uang ke kamera. Tekan 'q' untuk keluar.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Kamera gak ngerespon, bro!")
            break

        # AI memprediksi gambar dari webcam. 
        # conf=0.75 artinya AI cuma mau ngerespon kalau dia yakin di atas 75% (biar gak labil/salah sebut)
        # AI memprediksi gambar dari webcam.
        # TAMBAHIN workers=0 biar dia hemat RAM dan gak manggil shm.dll secara barbar
        results = model.predict(frame, conf=0.75, verbose=False, workers=0)
        
        # Ngegambar kotak bounding box hasil deteksi ke frame video asli
        frame_ada_kotak = results[0].plot()

        # LOGIKA SUARA: Cek apakah ada objek/uang yang tertangkap di kamera
        if len(results[0].boxes) > 0:
            
            # Ambil objek pertama yang tingkat keyakinannya paling tinggi
            box_pertama = results[0].boxes[0]
            id_kelas = int(box_pertama.cls[0].item()) # Ngambil ID kelas (angka 0, 1, 2, dst)
            nama_nominal = model.names[id_kelas]       # Nyocokin ID ke nama aslinya (misal: '100rb')

            # Cek waktu, apakah jeda 3 detiknya udah terpenuhi? (Biar gak spam suara)
            waktu_sekarang = time.time()
            if (waktu_sekarang - waktu_terakhir_ngomong) > jeda_suara:
                
                # Buat kalimat teksnya
                kalimat = f"Uang {nama_nominal} Rupiah"
                print(f"AI Mendeteksi: {kalimat}")
                
                # PANGGIL THREADING: Suruh laptop ngomong di "jalur belakang" 
                # args=(kalimat,) itu cara ngirim teksnya ke dalam fungsi ngomong
                thread_suara = threading.Thread(target=fungsi_ngomong, args=(kalimat,))
                thread_suara.start()
                
                # Catat waktu terakhir AI ngomong buat patokan jeda berikutnya
                waktu_terakhir_ngomong = waktu_sekarang

        # Tampilkan jendela video di layar laptop lo
        cv2.imshow("PI ARIF - DETEKSI UANG RUPIAH", frame_ada_kotak)

        # Kalau lo teken tombol 'q' di keyboard, aplikasi bakal ketutup rapi
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersih-bersih memori biar laptop gak lag pas program dimatiin
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
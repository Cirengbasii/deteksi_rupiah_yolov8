from ultralytics import YOLO

def main():
    print("Manggil model AI dasar...")
    model = YOLO('yolov8n.pt') 
    
    print("Mulai masuk kelas training pakai GPU di Drive D...")
    model.train(
        data=r'D:\PROJECT SENDIRI\BISMILLAHIRAHMANNIRRAHIM PI ARIF MUHAMMAD AKBAR\PI_Arif\uang-2016-2022-1\data.yaml', 
        epochs=100,          # Tetep gas settingan dendam lo wkwk
        imgsz=640,           
        batch=16,            # Pake 16 biar RTX 3050 lo kerja optimal & stabil
        device=0,            
        patience=30,         # Kasih napas 30 epoch buat nyari akurasi terbaik
        
        # === INI PARAMETER SAKTI BIAR PINDAH KE DRIVE D ===
        project=r'D:\PROJECT SENDIRI\BISMILLAHIRAHMANNIRRAHIM PI ARIF MUHAMMAD AKBAR\PI_Arif\runs',
        name='model_V4_TERBARUU' # Nama folder dalemnya
    )
    
    print("Proses training kelar, cek folder runs di Drive D lo!")

if __name__ == '__main__':
    main()
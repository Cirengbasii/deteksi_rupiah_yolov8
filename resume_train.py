from ultralytics import YOLO

def main():
    # Hapus kata 'detect' kalau emang foldernya langsung di dalam runs
    model = YOLO("runs/model_V4_TERBARUU/weights/last.pt")

    # Lanjutkan training
    model.train(resume=True)

if __name__ == '__main__':
    main()
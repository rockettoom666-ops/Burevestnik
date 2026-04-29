import multiprocessing

if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO("best.pt")
    model.train(
        data="/home/artem/Documents/TechnoStrelka/combined_dataset/data.yaml",
        epochs=25,
        imgsz=640,
        batch=20,
        lr0=0.001,
        freeze=10,
        cache=False,
        workers=0,
        name="final_combined",
        device=0,
        patience=5,
        cos_lr=True,
    )
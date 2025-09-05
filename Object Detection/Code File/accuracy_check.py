if __name__ == "__main__":
    from ultralytics import YOLO
    import numpy as np
    import multiprocessing

    multiprocessing.freeze_support()  # <— important on Windows

    # 1. Load model
    model = YOLO("waste_detection/baseline_train4/weights/best.pt")

    # 2. Run validation (this builds a DataLoader under the hood)
    metrics = model.val()  # returns a DetMetrics object

    # 3. Extract per-class AP@0.5
    per_class_ap50 = metrics.box.map.cpu().numpy()
    names = model.names

    # 4. Sort and print
    results = [(names[i], float(per_class_ap50[i])) for i in range(len(names))]
    results.sort(key=lambda x: x[1])
    print("Class Performance (AP@0.5) lowest → highest:")
    for cls, ap in results:
        print(f"{cls:30s}: {ap:.3f}")

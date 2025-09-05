import os

# Path to your labels folder
label_dirs = [
    "YOLO-Waste-Detection-1\YOLO-Waste-Detection-1/train\labels",
    "YOLO-Waste-Detection-1\YOLO-Waste-Detection-1/valid\labels",
    "YOLO-Waste-Detection-1\YOLO-Waste-Detection-1/test\labels",
]

# Your declared number of classes
NC = 43  # valid IDs 0 through 42

for ld in label_dirs:
    for fname in os.listdir(ld):
        path = os.path.join(ld, fname)
        with open(path) as f:
            lines = f.readlines()
        # If any line starts with “43 ” or higher, delete the file
        if any(int(line.split()[0]) >= NC for line in lines):
            print(f"Removing corrupt label: {path}")
            os.remove(path)

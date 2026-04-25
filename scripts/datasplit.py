import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = "dataset"  # original dataset
output_dir = "split_dataset"

classes = ["Tomato_Early_blight", "Tomato_Late_blight", "Tomato_healthy"]

for cls in classes:
    path = os.path.join(base_dir, cls)
    images = os.listdir(path)

    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    for split, data in zip(["train", "val", "test"], [train, val, test]):
        split_path = os.path.join(output_dir, split, cls)
        os.makedirs(split_path, exist_ok=True)

        for img in data:
            shutil.copy(os.path.join(path, img), os.path.join(split_path, img))
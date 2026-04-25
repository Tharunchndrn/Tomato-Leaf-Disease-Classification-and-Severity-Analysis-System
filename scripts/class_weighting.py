import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(train_dir):
    class_names = sorted([
        folder for folder in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, folder))
    ])

    labels = []

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(train_dir, class_name)
        image_count = len([
            f for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ])
        labels.extend([idx] * image_count)

    labels = np.array(labels)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )

    class_weights_dict = dict(enumerate(class_weights))

    return class_names, class_weights_dict
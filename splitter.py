import os
import shutil
import random

def split_dataset(source_folder, train_folder, test_folder, train_ratio=0.8):
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    for class_name in os.listdir(source_folder):
        class_path = os.path.join(source_folder, class_name)
        if os.path.isdir(class_path):
            print(f"Klasa: {class_name}")
            
            train_class_path = os.path.join(train_folder, class_name)
            test_class_path = os.path.join(test_folder, class_name)
            os.makedirs(train_class_path, exist_ok=True)
            os.makedirs(test_class_path, exist_ok=True)

            image_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(image_files)

            split_index = int(len(image_files) * train_ratio)
            train_files = image_files[:split_index]
            test_files = image_files[split_index:]

            for file_name in train_files:
                shutil.copy(os.path.join(class_path, file_name), train_class_path)
            for file_name in test_files:
                shutil.copy(os.path.join(class_path, file_name), test_class_path)
    
    print("Uspješno podijeljen dataset")

source_folder = "./dataset2"
train_folder = "./dataset2split/train"
test_folder = "./dataset2split/test"
split_dataset(source_folder, train_folder, test_folder, train_ratio=0.8)

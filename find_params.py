import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import GridSearchCV

def process_image(img_path, image_size=(96, 96)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, image_size)
        return img.flatten()
    return None

def load_images_from_folder_parallel(folder_path, image_size=(96, 96), max_images_per_class=100):
    features = []
    labels = []
    class_names = sorted(os.listdir(folder_path))

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            print(f"Učitavanje slika iz klase: {class_name}")
            image_paths = [os.path.join(class_folder, fname) for fname in os.listdir(class_folder)]
            image_paths = image_paths[:max_images_per_class]

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(lambda p: process_image(p, image_size), image_paths))

            for res in results:
                if res is not None:
                    features.append(res)
                    labels.append(label)
    
    print(f"Ukupan broj slika: {len(features)}")
    return np.array(features), np.array(labels), class_names

train_folder = "./dataset2split/train"
test_folder = "./dataset2split/test"

print("Učitavanje dataseta za treniranje")
features_train, labels_train, class_names = load_images_from_folder_parallel(train_folder)

print("Učitavanje dataseta za testiranje")
features_test, labels_test, _ = load_images_from_folder_parallel(test_folder)

print(f"Broj slika za treniranje: {len(features_train)}, Broj slika za testiranje: {len(features_test)}")
print(f"Klase: {class_names}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'C': [0.1, 1, 10, 50, 100],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=5)

grid.fit(X_train, y_train)

print("Najbolji parametri: ", grid.best_params_)
print("Najbolji estimator: ", grid.best_estimator_)
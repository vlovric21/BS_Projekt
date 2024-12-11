import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from concurrent.futures import ThreadPoolExecutor

def process_image(img_path, image_size=(48, 48)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, image_size)
        return img.flatten()
    return None

def load_images_from_folder_parallel(folder_path, image_size=(96, 96)):
    features = []
    labels = []
    class_names = sorted(os.listdir(folder_path))

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            print(f"Učitavanje slika iz klase: {class_name}")
            image_paths = [os.path.join(class_folder, fname) for fname in os.listdir(class_folder)]

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
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

print("Treniranje modela")
svm_model = SVC(kernel='rbf', C=10, gamma=0.0001, random_state=42)
svm_model.fit(features_train, labels_train)

joblib.dump(svm_model, "svm_ideal_params.pkl")
joblib.dump(scaler, "scaler_ideal_params.pkl")

print("Spremljeno")

print("Evaluacija modela")
predicted_labels = svm_model.predict(features_test)

print("\nIzvješće:")
print(classification_report(labels_test, predicted_labels, target_names=class_names))

conf_matrix = confusion_matrix(labels_test, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

accuracy = accuracy_score(labels_test, predicted_labels)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

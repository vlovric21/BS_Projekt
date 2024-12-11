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
from sklearn.model_selection import GridSearchCV

# 1. Parallelized function to process a single image
def process_image(img_path, image_size=(96, 96)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if img is not None:
        img = cv2.resize(img, image_size)  # Resize to the target size
        return img.flatten()
    return None

# 2. Parallel image loader
def load_images_from_folder_parallel(folder_path, image_size=(96, 96), max_images_per_class=100):
    X = []  # Features
    y = []  # Labels
    class_names = sorted(os.listdir(folder_path))  # Assume subfolders are named after classes

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            print(f"Loading images for class: {class_name}")
            image_paths = [os.path.join(class_folder, fname) for fname in os.listdir(class_folder)]
            image_paths = image_paths[:max_images_per_class]  # Limit to max_images_per_class

            # Use ThreadPoolExecutor for parallel loading
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(lambda p: process_image(p, image_size), image_paths))

            # Filter out None values (failed reads)
            for res in results:
                if res is not None:
                    X.append(res)
                    y.append(label)
    
    print(f"Total images loaded: {len(X)}")
    return np.array(X), np.array(y), class_names

# 3. Load datasets
train_folder = "./dataset2split/train"  # Adjust to your path
test_folder = "./dataset2split/test"  # Adjust to your path

print("Loading training set...")
X_train, y_train, class_names = load_images_from_folder_parallel(train_folder)

print("Loading testing set...")
X_test, y_test, _ = load_images_from_folder_parallel(test_folder)

print(f"Number of training samples: {len(X_train)}, Number of testing samples: {len(X_test)}")
print(f"Classes: {class_names}")

# 4. Feature Scaling (multi-core)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 50, 100],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

# 6. Initialize the GridSearchCV object
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=5)

# 7. Fit the grid search to the data
grid.fit(X_train, y_train)

# 8. Print the best parameters and the best estimator
print("Najbolji parametri: ", grid.best_params_)
print("Najbolji estimator: ", grid.best_estimator_)

'''
# 9. Evaluate the best model on the test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

'''
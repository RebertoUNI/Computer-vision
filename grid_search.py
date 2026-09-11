import cv2
import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from scipy.spatial.distance import cdist
from sklearn.multiclass import OutputCodeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.extract_features import read_split

# ==============================================================================
# HYPERPARAMETER GRID
# ==============================================================================
# Modify these lists to test different combinations.
# Note: execution time grows exponentially with the number of combinations!
GRID = {
    'N_FEATURES': [150],
    'DENSE': [True],
    'NUM_SAMPLES': [100000],
    'NUM_CLUSTER': [80],
    'SPATIAL_PYRAMID': [False, True],
    'SOFT_ASSIGNMENT': [False, True],
    'BETA': [0.01],
    'K_NEIGHBORS': [7]
}

# Output directories
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# HELPER FUNCTIONS (From project.ipynb)
# ==============================================================================
def generate_dense_keypoints(image_shape, step_size=15, scales=[10, 20]):
    keypoints = []
    rows, cols = image_shape
    for y in range(0, rows, step_size):
        for x in range(0, cols, step_size):
            for size in scales:
                kp = cv2.KeyPoint(float(x), float(y), float(size))
                keypoints.append(kp)
    return keypoints

def extract_sift_for_images(image_paths, labels, sift_detector, dense=False):
    keypoints_list = []
    descriptors_list = []
    valid_labels = []

    for image_path, label in zip(image_paths, labels):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Cannot read image: {image_path}")
            continue

        if dense:
            dense_kp = generate_dense_keypoints(image.shape, step_size=15, scales=[10, 20])
            keypoints, descriptors = sift_detector.compute(image, dense_kp)
        else:
            keypoints, descriptors = sift_detector.detectAndCompute(image, None) 

        keypoints_list.append(keypoints)
        if descriptors is not None:
            descriptors_list.append(descriptors)
        else:
            descriptors_list.append(np.empty((0, 128)))
        valid_labels.append(label)

    return keypoints_list, descriptors_list, np.array(valid_labels)

def compute_histograms(
    image_paths, 
    keypoints_list, 
    descriptors_list, 
    kmeans_model, 
    dense, 
    spatial_pyramid, 
    soft_assignment, 
    beta
):
    num_clusters = kmeans_model.n_clusters
    total_bins = num_clusters * 21 if spatial_pyramid else num_clusters 

    X = []
    for image_path, keypoints, descriptors in zip(image_paths, keypoints_list, descriptors_list):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        h, w = image.shape

        if descriptors is None or len(descriptors) == 0:
            X.append(np.zeros(total_bins))
            continue

        if not soft_assignment:
            words = kmeans_model.predict(descriptors)
        else:
            centroids = kmeans_model.cluster_centers_
            distances_sq = cdist(descriptors, centroids, metric='sqeuclidean')
            weights = np.exp(-beta * distances_sq)
            row_sums = np.sum(weights, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0 
            weights = weights / row_sums 

        if not spatial_pyramid:
            if not soft_assignment:
                hist, _ = np.histogram(words, bins=num_clusters, range=(0, num_clusters))
                hist = hist.astype(np.float64)
            else:
                hist = np.sum(weights, axis=0)

            total_mass = np.sum(hist)
            if total_mass > 0:
                hist = hist / total_mass
            final_hist = hist
        else:
            hist_L0 = np.zeros(num_clusters)
            hist_L1 = np.zeros((4, num_clusters))
            hist_L2 = np.zeros((16, num_clusters))

            if not soft_assignment:
                for kp, word in zip(keypoints, words):
                    x, y = kp.pt
                    hist_L0[word] += 1
                    col_L1 = min(int((x / w) * 2), 1)
                    row_L1 = min(int((y / h) * 2), 1)
                    hist_L1[row_L1 * 2 + col_L1, word] += 1
                    col_L2 = min(int((x / w) * 4), 3)
                    row_L2 = min(int((y / h) * 4), 3)
                    hist_L2[row_L2 * 4 + col_L2, word] += 1
            else:
                for kp, w_vec in zip(keypoints, weights):
                    x, y = kp.pt
                    hist_L0 += w_vec
                    col_L1 = min(int((x / w) * 2), 1)
                    row_L1 = min(int((y / h) * 2), 1)
                    hist_L1[row_L1 * 2 + col_L1] += w_vec
                    col_L2 = min(int((x / w) * 4), 3)
                    row_L2 = min(int((y / h) * 4), 3)
                    hist_L2[row_L2 * 4 + col_L2] += w_vec

            hist_L0 *= 0.25
            hist_L1 *= 0.25
            hist_L2 *= 0.50

            hists_to_stack = ([hist_L0] + [hist_L1[i] for i in range(4)] + [hist_L2[i] for i in range(16)])
            final_hist = np.concatenate(hists_to_stack)

            norm = np.linalg.norm(final_hist)
            if norm > 0:
                final_hist = final_hist / norm
                
        X.append(final_hist)
    return np.array(X)

def chi2_kernel(X, Y, gamma=1.0):
    result = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            denom = x + y
            denom[denom == 0] = 1e-10  # evita div per zero
            result[i, j] = np.exp(-gamma * np.sum((x - y)**2 / denom))
    return result

def save_confusion_matrix(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("Loading datasets...")
    train_path = Path("dataset") / "train"
    test_path = Path("dataset") / "test"
    
    # Check if dataset exists, to fail early gracefully if not
    if not train_path.exists() or not test_path.exists():
        print("Error: dataset/train or dataset/test does not exist. Run from correct directory.")
        return

    train_paths, train_label_ids, classes = read_split(train_path)
    test_paths, test_label_ids, test_classes = read_split(test_path)

    if classes != test_classes:
        raise ValueError(f"Train and test classes do not match: {classes} != {test_classes}")

    train_labels = np.array([classes[label_id] for label_id in train_label_ids])
    test_labels = np.array([classes[label_id] for label_id in test_label_ids])

    print(f"Found {len(train_paths)} training images.")
    print(f"Found {len(test_paths)} test images.")
    
    # Store results
    results = []
    
    # To efficiently execute the grid search, we use nested loops so that intermediate 
    # steps like SIFT extraction and K-Means are not redundantly re-computed.
    for n_features in GRID['N_FEATURES']:
        sift = cv2.SIFT_create(nfeatures=n_features)
        
        for dense in GRID['DENSE']:
            print(f"\\n--- EXTRACTING SIFT: N_FEATURES={n_features}, DENSE={dense} ---")
            train_kp, train_desc, train_y = extract_sift_for_images(train_paths, train_labels, sift, dense)
            test_kp, test_desc, test_y = extract_sift_for_images(test_paths, test_labels, sift, dense)
            
            # Combine all training descriptors for K-Means sampling
            stacked_descriptors = [d for d in train_desc if len(d) > 0]
            if len(stacked_descriptors) > 0:
                stacked_descriptors = np.vstack(stacked_descriptors)
            else:
                stacked_descriptors = np.empty((0, 128))
            total_descriptors = stacked_descriptors.shape[0]
            
            for num_samples in GRID['NUM_SAMPLES']:
                # Sample descriptors for K-means
                if total_descriptors > num_samples:
                    np.random.seed(42)
                    random_indices = np.random.choice(total_descriptors, num_samples, replace=False)
                    sampled_descriptors = stacked_descriptors[random_indices, :]
                else:
                    sampled_descriptors = stacked_descriptors
                    
                for num_cluster in GRID['NUM_CLUSTER']:
                    print(f"\\n--- K-MEANS: NUM_CLUSTER={num_cluster}, NUM_SAMPLES={num_samples} ---")
                    if sampled_descriptors.shape[0] == 0:
                        print("No descriptors found. Skipping...")
                        continue
                        
                    kmeans = KMeans(n_clusters=num_cluster, random_state=42, n_init=10)
                    kmeans.fit(sampled_descriptors)
                    
                    for sp in GRID['SPATIAL_PYRAMID']:
                        for sa in GRID['SOFT_ASSIGNMENT']:
                            for beta in GRID['BETA']:
                                print(f"--- HISTOGRAMS: SPATIAL_PYRAMID={sp}, SOFT_ASSIGNMENT={sa}, BETA={beta} ---")
                                X_train = compute_histograms(
                                    train_paths, train_kp, train_desc, kmeans, 
                                    dense, sp, sa, beta
                                )
                                X_test = compute_histograms(
                                    test_paths, test_kp, test_desc, kmeans, 
                                    dense, sp, sa, beta
                                )
                                
                                for k_neighbors in GRID['K_NEIGHBORS']:
                                    config_str = (
                                        f"NF{n_features}_D{dense}_NS{num_samples}_NC{num_cluster}_"
                                        f"SP{sp}_SA{sa}_B{beta}_K{k_neighbors}"
                                    )
                                    print(f"\\nEvaluating Config: {config_str}")
                                    
                                    # 1. K-NN
                                    nn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='euclidean')
                                    nn.fit(X_train, train_y)
                                    y_pred_nn = nn.predict(X_test)
                                    acc_nn = accuracy_score(test_y, y_pred_nn)
                                    
                                    # Save K-NN Confusion Matrix
                                    save_confusion_matrix(
                                        test_y, y_pred_nn, classes, 
                                        f"KNN (K={k_neighbors}) - Acc: {acc_nn:.2%}",
                                        FIGURES_DIR / f"knn_{config_str}.png"
                                    )
                                    
                                    # 2. Linear SVM
                                    svm_linear = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)
                                    svm_linear.fit(X_train, train_y)
                                    y_pred_linear = svm_linear.predict(X_test)
                                    acc_linear = accuracy_score(test_y, y_pred_linear)
                                    
                                    # Save Linear SVM Confusion Matrix
                                    save_confusion_matrix(
                                        test_y, y_pred_linear, classes, 
                                        f"Linear SVM - Acc: {acc_linear:.2%}",
                                        FIGURES_DIR / f"linear_svm_{config_str}.png"
                                    )
                                    
                                    # 3. Chi-Squared SVM
                                    svm_chi2 = SVC(kernel=lambda X, Y: chi2_kernel(X, Y, gamma=0.5))
                                    svm_chi2.fit(X_train, train_y)
                                    y_pred_chi2 = svm_chi2.predict(X_test)
                                    acc_chi2 = accuracy_score(test_y, y_pred_chi2)
                                    
                                    # Save Chi-Squared SVM Confusion Matrix
                                    save_confusion_matrix(
                                        test_y, y_pred_chi2, classes, 
                                        f"Chi-Squared SVM - Acc: {acc_chi2:.2%}",
                                        FIGURES_DIR / f"chi2_svm_{config_str}.png"
                                    )
                                    
                                    # 4. ECOC-SVM
                                    base_svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
                                    ecoc = make_pipeline(
                                        StandardScaler(),
                                        OutputCodeClassifier(estimator=base_svm, code_size=1.5, random_state=42),
                                    )
                                    ecoc.fit(X_train, train_y)
                                    y_pred_ecoc = ecoc.predict(X_test)
                                    acc_ecoc = accuracy_score(test_y, y_pred_ecoc)
                                    
                                    # Save ECOC-SVM Confusion Matrix
                                    save_confusion_matrix(
                                        test_y, y_pred_ecoc, classes, 
                                        f"ECOC-SVM - Acc: {acc_ecoc:.2%}",
                                        FIGURES_DIR / f"ecoc_{config_str}.png"
                                    )
                                    
                                    print(f"KNN Accuracy: {acc_nn:.4f} | Linear SVM: {acc_linear:.4f} | Chi2 SVM: {acc_chi2:.4f} | ECOC Accuracy: {acc_ecoc:.4f}")
                                    
                                    # Log Results
                                    results.append({
                                        'N_FEATURES': n_features,
                                        'DENSE': dense,
                                        'NUM_SAMPLES': num_samples,
                                        'NUM_CLUSTER': num_cluster,
                                        'SPATIAL_PYRAMID': sp,
                                        'SOFT_ASSIGNMENT': sa,
                                        'BETA': beta,
                                        'K_NEIGHBORS': k_neighbors,
                                        'KNN_ACCURACY': acc_nn,
                                        'LINEAR_SVM_ACCURACY': acc_linear,
                                        'CHI2_SVM_ACCURACY': acc_chi2,
                                        'ECOC_SVM_ACCURACY': acc_ecoc
                                    })
                                    
                                    # Save intermediate results in case script is stopped early
                                    pd.DataFrame(results).to_csv(RESULTS_DIR / "grid_search_results.csv", index=False)

    print("\\nGrid search completed! Results saved to ./results/grid_search_results.csv")
    
if __name__ == "__main__":
    main()

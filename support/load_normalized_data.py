# Loads all .pgm images from the AT&T dataset, flattens and normalises them.
# Returns X (image matrix) and y (labels/subject IDs)
# Saves arrays to data/processed/ for faster future loading
# Run from repo root: python -m support.load_data

import os           # for file and directory operations
import cv2          # for image processing (OpenCV library)
import numpy as np  # for numerical operations (NumPy library)

def load_dataset(data_dir=None):
    # BASE_DIR is always needed (for processed_dir), so move it outside the if block
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if data_dir is None:
        data_dir = os.path.join(BASE_DIR, "data", "raw")                # default path to dataset, can be overridden by argument

    # --- Folder where arrays will be saved ---
    processed_dir = os.path.join(BASE_DIR, "data", "processed")         # create this folder if it doesn't exist, to store the processed .npy files for faster loading next time
    os.makedirs(processed_dir, exist_ok=True)

    # Paths for the saved .npy files
    flattened_img_path = os.path.join(processed_dir, "flattened_img.npy") 
    subject_ids_path   = os.path.join(processed_dir, "subject_ids.npy")

    # --- If already saved, just load them ---
    if os.path.exists(flattened_img_path) and os.path.exists(subject_ids_path):
        print("Found saved arrays, loading from data/processed/...")
        flattened_img = np.load(flattened_img_path)
        subject_ids   = np.load(subject_ids_path)
        print(f"Loaded {flattened_img.shape[0]} images, {flattened_img.shape[1]} features each.")
        print(f"Subjects found: {len(np.unique(subject_ids))}")
        return flattened_img, subject_ids

    # --- Otherwise read all images from scratch ---
    print("No saved arrays found, reading images from data/...")

    flattened_img = []  # list to hold flattened image vectors
    subject_ids   = []  # list to hold corresponding subject IDs

    # Loop through each subject folder (s1, s2, ..., s40)
    for subject_folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, subject_folder)

        if not os.path.isdir(folder_path):
            continue

        if not subject_folder.startswith("s"):
            continue

        # Extract subject ID number from folder name e.g. "s1" -> 1
        subject_id = int(subject_folder[1:])

        # Loop through each .pgm image in the subject folder
        for img_file in sorted(os.listdir(folder_path)):
            if not img_file.endswith(".pgm"):
                continue
            
            # Construct full path to the image file and read it in grayscale mode
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # shape: (112, 92)

            # Check if the image was read successfully
            if img is None:
                print(f"Warning: could not read {img_path}")
                continue

            # Flatten 92x112 image to 1D vector of 10304 features
            img_flat = img.flatten()

            # Normalise pixel values from 0-255 to 0.0-1.0
            img_normalised = img_flat / 255.0

            # Append the normalised, flattened image and its subject ID to the respective lists
            flattened_img.append(img_normalised)
            subject_ids.append(subject_id)

    flattened_img = np.array(flattened_img)  # Matrix: (400 X 10304)
    subject_ids   = np.array(subject_ids)    # 1D Array: (400)

    # --- Save arrays to data/processed/ ---
    np.save(flattened_img_path, flattened_img)
    np.save(subject_ids_path, subject_ids)
    print("Arrays saved to data/processed/")

    print(f"Loaded {flattened_img.shape[0]} images, {flattened_img.shape[1]} features each.")
    print(f"Subjects found: {len(np.unique(subject_ids))}")

    return flattened_img, subject_ids


if __name__ == "__main__":
    X, y = load_dataset()
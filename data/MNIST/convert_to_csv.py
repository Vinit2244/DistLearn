import gzip
import numpy as np
import csv
from pathlib import Path

def read_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        _ = f.read(16)  # skip magic number and dimensions
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28 * 28)
    return images

def read_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        _ = f.read(8)  # skip magic number and item count
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Set paths
raw_dir = Path('./raw')  # Adjust this if your raw files are elsewhere
img_path = raw_dir / 't10k-images-idx3-ubyte.gz'
lbl_path = raw_dir / 't10k-labels-idx1-ubyte.gz'

# Load data
images = read_images(img_path)
labels = read_labels(lbl_path)

# Output CSV path
output_csv = '../mnist_test.csv'

# Write to CSV
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Optional header
    header = [f'pixel{i}' for i in range(784)] + ['label']
    writer.writerow(header)
    # Write data
    for label, image in zip(labels, images):
        writer.writerow(image.tolist() + [label])

print(f"Saved test set to {output_csv}")

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os

# Function to compress an image using Singular Value Decomposition (SVD)
def img_compress(image_path, k):  
    # Load the image
    image = io.imread(image_path)
    
    # Convert the image to grayscale
    gray_img = color.rgb2gray(image)
    
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(gray_img, full_matrices=False)
    
    # Create a diagonal matrix with the top k singular values
    truncated_S = np.zeros((k, k))
    np.fill_diagonal(truncated_S, S[:k])
    
    # Truncate U and Vt matrices to keep only the top k components
    truncated_U = U[:, :k]
    truncated_Vt = Vt[:k, :]
    
    # Reconstruct the compressed image using the truncated matrices
    compressed_image = np.dot(np.dot(truncated_U, truncated_S), truncated_Vt)
    
    # Compute the original image size in bytes
    original_size = os.path.getsize(image_path)
    
    # Estimate the compressed size using SVD components
    compressed_size = k * (U.shape[0] + Vt.shape[1] + 1) * 8 / (U.shape[0] * Vt.shape[1])
    
    # Display the original and compressed images
    plt.figure(figsize=(20, 20))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(gray_img, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Compressed Image")
    plt.imshow(compressed_image, cmap='gray')
    
    plt.show()

    # Print compression details
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {original_size / compressed_size}")
    print(f"Space saving: {original_size - compressed_size} bytes")

# Get user input for the image path and the value of k
image = input("Enter the path of the image: ")
k = int(input("Enter the value of k: "))

# Call the function to compress the image
img_compress(image, k)

#C:\Users\HP\Desktop\myProjects\python\images\fist-bump.jpg
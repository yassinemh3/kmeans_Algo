import numpy as np
import matplotlib.pyplot as plt


def initialize_centroids(data, k):
    """Randomly initializes the centroids from the data points."""
    indices = np.random.permutation(len(data))[:k]
    centroids = data[indices]
    return centroids


def assign_clusters(data, centroids):
    """Assigns each data point to the nearest centroid using Manhattan distance."""
    distances = np.abs(data - centroids[:, np.newaxis]).sum(axis=2)
    # """Assigns each data point to the nearest centroid using Euclidean distance"""
    # distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def update_centroids(data, assignments, k):
    """Assigns each data point to the nearest centroid."""
    new_centroids = np.array([data[assignments == i].mean(axis=0) for i in range(k)])
    return new_centroids


def k_means(data, k, max_iterations=100):
    """Performs k-means clustering."""
    # Step 1: Initialize centroids randomly from the data points
    centroids = initialize_centroids(data, k)
    initial_centroids = centroids.copy()  # Store initial centroids for plotting
    print(centroids)
    for _ in range(max_iterations):
        # Step 2: Assign points to the nearest centroid
        assignments = assign_clusters(data, centroids)

        # Step 3: Update centroids to the mean of the points assigned to them
        new_centroids = update_centroids(data, assignments, k)

        # Check for convergence (if centroids do not change)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, assignments, initial_centroids


# Example usage:
# Generating some random data
data = np.random.rand(300, 2)  # 300 points in 2D space
k = 2  # Number of clusters

centroids, assignments, initial_centroids = k_means(data, k)
print(data)
print("Centroids:\n", centroids)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap='viridis', marker='o', label='Data points')
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='blue', s=200, alpha=0.5, marker='o', label='Initial Centroids')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Final Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

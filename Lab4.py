#!/usr/bin/env python
# coding: utf-8

# In[42]:


from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate


# In[43]:


# Load the Olivetti faces dataset
data = fetch_olivetti_faces()
images = data.images
X = data.data
y = data.target


# In[44]:


# Initial split to get training + validation and test set (20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Further split training + validation set into training and validation set (20% validation from remaining data)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)


# In[45]:


# Apply PCA on training data, preserving 99% variance
pca = PCA(n_components=0.99, random_state=42)
X_train_pca = pca.fit_transform(X_train)

# Apply the same PCA transformation to validation and test sets
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)


# In[46]:


# Define covariance types and an empty dictionary to store BIC scores
covariance_types = ['full', 'tied', 'diag', 'spherical']
bic_scores_co = {}

# Fit GMM with different covariance types and calculate BIC scores
for cov_type in covariance_types:
    gmm = GaussianMixture(n_components=10, covariance_type=cov_type, random_state=42)
    gmm.fit(X_train_pca)
    bic_scores_co[cov_type] = gmm.bic(X_val_pca)

# Select the covariance type with the lowest BIC score
best_cov_type = min(bic_scores_co, key=bic_scores_co.get)
print(best_cov_type)


# In[47]:


# Range of clusters to try
cluster_range = range(1, 21)
aic_scores = []
bic_scores = []

# Fit GMM with different numbers of clusters
for n_clusters in cluster_range:
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=best_cov_type, random_state=42)
    gmm.fit(X_train_pca)
    aic_scores.append(gmm.aic(X_val_pca))
    bic_scores.append(gmm.bic(X_val_pca))

# Determine the optimal number of clusters based on the lowest BIC score
optimal_clusters = cluster_range[np.argmin(bic_scores)]
print(optimal_clusters)


# In[48]:


# Plot PCA reduced data (only the first two components for visualization)
plt.figure(figsize=(8, 8))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], s=10, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Reduced Data')
plt.show()


# In[49]:


# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.axhline(y=0.99, color='r', linestyle='--', label='99% variance')
plt.legend()
plt.show()


# In[50]:


# Plot BIC scores for each covariance type

scores = list(bic_scores_co.values())
plt.figure(figsize=(8, 6))
plt.bar(covariance_types, scores, color='skyblue')
plt.yscale('log')  # Apply logarithmic scale to y-axis
plt.xlabel('Covariance Type')
plt.ylabel('BIC Score (Log Scale)')
plt.title('BIC Scores for Different Covariance Types')
plt.show()


# In[51]:


# Plot AIC and BIC scores
plt.figure(figsize=(12, 6))
plt.plot(cluster_range, aic_scores, label='AIC Score', marker='o')
plt.plot(cluster_range, bic_scores, label='BIC Score', marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('AIC and BIC Scores vs. Number of Clusters')
plt.legend()
plt.show()


# In[52]:


# Fit GMM with optimal number of clusters and best covariance type on training data
gmm = GaussianMixture(n_components=optimal_clusters, covariance_type=best_cov_type, random_state=42)
gmm.fit(X_train_pca)

# Output hard clustering assignments for each instance in the validation set
hard_assignments = gmm.predict(X_val_pca)
print("Hard Clustering Assignments:\n", hard_assignments)


# In[53]:


# Output soft clustering probabilities for each instance in the validation set
soft_probabilities = gmm.predict_proba(X_val_pca)
print("Soft Clustering Probabilities:\n", soft_probabilities)


# In[54]:


# Generate new samples from the GMM (10 samples, for example)
generated_samples, _ = gmm.sample(10)

# Transform samples back to the original space using the inverse of the PCA
generated_faces = pca.inverse_transform(generated_samples)

# Visualize the generated faces
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_faces[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
plt.suptitle("Generated Faces")
plt.show()


# In[55]:


# Select a few images to modify
original_images = X_val[:3]  # Select first three images for modification

# Create transformations
rotated_images = [rotate(img.reshape(64, 64), angle=45, mode='wrap').flatten() for img in original_images]  # Rotate by 45 degrees
flipped_images = [np.fliplr(img.reshape(64, 64)).flatten() for img in original_images]  # Horizontal flip
darkened_images = [np.clip(img * 0.5, 0, 1).flatten() for img in original_images]  # Darken by reducing brightness

# Visualize modified images
plt.figure(figsize=(9, 3))
for i, (orig, rotated, flipped, darkened) in enumerate(zip(original_images, rotated_images, flipped_images, darkened_images)):
    plt.subplot(3, 4, i*4 + 1)
    plt.imshow(orig.reshape(64, 64), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(3, 4, i*4 + 2)
    plt.imshow(rotated.reshape(64, 64), cmap='gray')
    plt.title("Rotated")
    plt.axis('off')
    
    plt.subplot(3, 4, i*4 + 3)
    plt.imshow(flipped.reshape(64, 64), cmap='gray')
    plt.title("Flipped")
    plt.axis('off')
    
    plt.subplot(3, 4, i*4 + 4)
    plt.imshow(darkened.reshape(64, 64), cmap='gray')
    plt.title("Darkened")
    plt.axis('off')
plt.suptitle("Modified Images")
plt.show()


# In[56]:


# Calculate log-likelihoods for original images
original_scores = gmm.score_samples(pca.transform(original_images))

# Calculate log-likelihoods for modified images
rotated_scores = gmm.score_samples(pca.transform(rotated_images))
flipped_scores = gmm.score_samples(pca.transform(flipped_images))
darkened_scores = gmm.score_samples(pca.transform(darkened_images))

# Print out the scores to observe differences
print("Log-likelihood scores for original images:", original_scores)
print("Log-likelihood scores for rotated images:", rotated_scores)
print("Log-likelihood scores for flipped images:", flipped_scores)
print("Log-likelihood scores for darkened images:", darkened_scores)


# In[ ]:





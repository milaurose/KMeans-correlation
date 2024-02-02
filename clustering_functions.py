#!/usr/bin/env python
# coding: utf-8
# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#define custom kmeans class and functions for clustering using correlation as a distance - similar format as sklearn kmeans but with the matlab functionality
class KMeansCorrelation:
    def __init__(self, n_clusters, n_rep=1, max_iters=300, tol = 1e-4, threshold = 5, random_state=None):
        self.n_clusters = n_clusters
        self.n_rep = n_rep #the number of replicates is num of times to repeat with new initial cluster positions (like n_init in scikit-learn)
        self.max_iters = max_iters #number of iterations in each replicate
        self.tol = tol #convergence tolerance (Frobenius norm between consecutive centroids)
        self.threshold = threshold
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def correlation_distance(self, X, Y):
        # Ensure X and Y have the same shape
        assert X.shape == Y.shape, "Input matrices must have the same shape"
        # Calculate correlation distance
        return 1 - np.corrcoef(X, Y, rowvar=False)[0, 1]

    def fit(self, data):
        best_labels, best_centroids, best_inertia = None, None, np.inf

        for rep in range(self.n_rep):
            # Randomly initialize centroids with a different seed for each repetition
            rng = np.random.default_rng(seed=self.random_state + rep) if self.random_state is not None else np.random.default_rng()
            centroids = data[rng.choice(data.shape[0], self.n_clusters, replace=False)]

            iter_count = 0 
            centers_squared_diff = 1
            #iteratively update the location of the centroid (from random) until the number of max iterations is reached, or centroid location barely changes (tolerance)
            while (centers_squared_diff > self.tol) and (iter_count < self.max_iters):
                # Assign each data point to the nearest centroid based on correlation distance
                distances = np.array([np.apply_along_axis(lambda x: self.correlation_distance(x, centroid), 1, data) for centroid in centroids])
                labels = np.argmin(distances, axis=0)

                # Update centroids based on the mean of assigned data points
                centroids_new = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])

                #compute the tolerance (how different the new centroids are compared to the previous iteration) using Frobenius norm
                centers_squared_diff = np.sum((centroids_new - centroids) ** 2)
                centroids = centroids_new
                
                #update the iteration number
                print(iter_count)
                iter_count = iter_count + 1
                
                #print the convergence situation
                if centers_squared_diff <= self.tol:
                    print(" Random repetition #" + str(rep) + " converged on iteration " + str(iter_count) + " as tolerance of " + str(centers_squared_diff) + " was reached.")
                elif iter_count == self.max_iters:
                    print(" Random repetition #" + str(rep) + " converged at max iteration " + str(iter_count) + " at tolerance of " + str(centers_squared_diff) + ".")
                else:
                    print("\r","initialization: " + str(rep) + " iteration: " + str(iter_count) + " tolerance: " + str(centers_squared_diff), end="")


            # Calculate within-cluster sums of point-to-centroid distances (inertia)
            inertia = np.sum([np.sum((data[labels == i] - centroids[i])**2) for i in range(self.n_clusters)])

            # Update best solution if the current one has lower inertia
            if inertia < best_inertia:
                best_labels, best_centroids, best_inertia = labels, centroids, inertia
        self.labels_ = best_labels
        self.cluster_centers_ = best_centroids
        return self

    def transform(self, data):
        #to get actual correlation, subtract 1 and divide by -1 from the correlation_distance
        correlations = np.array([np.apply_along_axis(lambda x: (self.correlation_distance(x, centroid)-1.0)/-1.0, 1, data) for centroid in self.cluster_centers_])
        return correlations.T

    def predict(self, data):
        correlations = self.transform(data)
        distances = 1-correlations
        prediction = np.argmin(distances, axis=1)
        print(self.threshold)
        
        #iterate over the clusters, find the timepoints within the lowest correlation percentile of each cluster, assign all these timepoints to a new cluster
        correlations_within_clust = pd.DataFrame(correlations).loc[prediction==0, 0]
        percentile_cutoff_value = np.percentile(correlations_within_clust,self.threshold, axis = 0)
        bool_corr_within_percentile = correlations_within_clust < percentile_cutoff_value

        #also plot the histograms of correlations within each cluster and the percentile
        fig,axs = plt.subplots(nrows=2, ncols=self.n_clusters, figsize=(self.n_clusters*4,7), dpi = 200)
        axs[1,0].hist(1-correlations_within_clust)
        axs[1,0].axvline(1-percentile_cutoff_value, color = 'red')
        axs[1,0].set_xlabel('Distances (1-r)')
        axs[1,0].set_ylabel('Number of timepoints')
        axs[0,0].set_title('Cluster 0 \n' + str(len(correlations_within_clust)) + ' timepoints \n Discard correlations below ' + str(round(percentile_cutoff_value, 4)))
        axs[0,0].hist(correlations_within_clust)
        axs[0,0].axvline(percentile_cutoff_value, color = 'red')
        axs[0,0].set_xlabel('Correlations (r)')
        axs[0,0].set_ylabel('Number of timepoints')

        for clust in range(1,self.n_clusters):
            correlations_within_clust_new = pd.DataFrame(correlations).loc[prediction==clust, clust]
            percentile_cutoff_value = np.percentile(correlations_within_clust_new,self.threshold, axis = 0)
            bool_corr_within_percentile_new = correlations_within_clust_new < percentile_cutoff_value
            
            #also plot the histograms of correlations within each cluster and the percentile
            axs[1,clust].hist(1-correlations_within_clust)
            axs[1,clust].axvline(1-percentile_cutoff_value, color = 'red')
            axs[1,clust].set_xlabel('Distances (1-r)')
            axs[0,clust].set_title('Cluster ' + str(clust) + '\n' + str(len(correlations_within_clust)) + ' timepoints \n Discard correlations below ' + str(round(percentile_cutoff_value, 4)))
            axs[0,clust].hist(correlations_within_clust)
            axs[0,clust].axvline(percentile_cutoff_value, color = 'red')
            axs[0,clust].set_xlabel('Correlations (r)')

            #concatenate results across clusters using index
            correlations_within_clust = pd.concat([correlations_within_clust, correlations_within_clust_new])
            bool_corr_within_percentile = pd.concat([bool_corr_within_percentile, bool_corr_within_percentile_new])

        #sort the final df so indices are in order
        bool_corr_within_percentile = bool_corr_within_percentile.sort_index()
        correlations_within_clust = correlations_within_clust.sort_index()

        #plot final figure
        fig.show()

        #where the dist is in lowest percentile, assign that timepoint to a new cluster
        prediction[bool_corr_within_percentile] = clust + 1
        return correlations_within_clust, prediction


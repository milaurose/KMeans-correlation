# KMeans-correlation

## Purpose
Performs kMeans clustering using correlation as a distance metric, intended specifically for use on fMRI data to retrieve co-activation patterns (CAPs). It mimics the matlab kmeans function when used with 'correlation' as the distance option (https://www.mathworks.com/help/stats/kmeans.html#buefs04-Distance) - this is the function used in the tbCAPs toolbox (https://github.com/FabienCarruzzo/tbCAPS). However, unlike the matlab kmeans function, the structure of inputs, outputs and attributes of our function mirror the scikitlearn kmeans function (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). 

Therefore, we provide a pythonic implementation of kmeans correlation-based clustering; a hybrid between the options available from matlab and scikit-learn. It is convenient for use on rodent fMRI data that has already been censored and corrected for confounds through RABIES, since the tbCAPs toolbox is primarily designed for use on human data prepared in a certain format.

### Additional features
A unique feature of our class compared to basic kmeans clustering algorithms is that the timepoints with the lowest x% of correlations assigned to each of n given clusters are 'discarded' (ie all assigned to cluster n+1). The percent threshold (x) determining how many timepoints are discarded can be specified with the option 'threshold'. This behaviour is intended to mirror the CAP_AssignFrames function from the tbCAPs toolbox (https://github.com/FabienCarruzzo/tbCAPS/blob/master/Analysis/CAP_AssignFrames.m). 

## Options
n_clusters: The number of clusters to extract.

n_rep: Number of times to repeat the clustering with different random initializations (similar to scikitlearn's n_init parameter). Default 1.

max_iters: maximum number of iterations per repetition. Default 300.

tol: relative tolerance to declare convergence. Helps speed up the process since convergence usually occurs before max_iters is reached. Default: 1e-4.

threshold: percent of timepoints with the lowest correlations within each cluster to discard. Default 5%.

random_state: an integer defined a seed for the random initialization of centroids, to allow for reproducible results across runs. Default None.

## Methods and examples

**fit**

``` kmeans_object = KMeansCorrelation(n_clusters=3, n_rep=10,  max_iters=300, random_state=0).fit(data) ```

Data is an array of timepoints by voxels. The fit method returns a kmeans_object. You can then access the cluster centers (array of n_clusters by n_voxels ie a map of each centroid) with ``` kmeans_object.cluster_centers_ ``` and labels (1D array with dimension n_timepoints containing the cluster number that each timepoint is assigned to) with ```kmeans_object.labels_```.

**transform**

```correlations = kmeans_object.transform(data)```

Transform the data to obtain the correlation of each timepoint to each cluster (array of n_timepoints by n_clusters). Note that no timepoints with low correlations are discarded here, only in 'predict'. The distance metric is 1-correlation.

**predict**

``` correlations_within_clust, labels_new = kmeans_object.predict(data_new)) ```

Intended to be applied on a new dataset, separate from the one that the original clustering was performed on. data_new should have the same number of voxels as the original dataset, but can have a different number of timepoints. Returns a 1D array (dimension n_timepoints) of the correlation between each timepoint and the cluster that it was assigned to. Also returns 1D array of the predicted label at each timepoint (if the timepoint had a low correlation to the cluster it was assigned to, it will be discarded and assigned to an 'extra' cluster). This method will also output a figure showing the histogram of correlations within each cluster.

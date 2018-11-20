import numpy as np


class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
             Finds n_cluster in the data x
             params:
                 x - N X D numpy array
             returns:
                 A tuple
                 (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
             Note: Number of iterations is the number of time you update the assignment
         '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        centroids = x[np.random.choice(N, self.n_cluster, replace=False), :]
        prevJ = 10e10
        J = 0
        K = self.n_cluster
        membership = []
        assignment = np.zeros((N, K))
        number_of_updates = 0
        for iteration in range(self.max_iter):
            J = 0
            membership = []
            assignment = np.zeros((N, K))
            for i in range(N):
                distance_to_center = []
                for j in range(K):
                    _distance = np.inner(x[i] - centroids[j],x[i] - centroids[j])
                    distance_to_center.append(_distance)
                belong = np.argmin(np.array(distance_to_center))
                membership.append(belong)
                assignment[i,belong] = 1
                J += np.inner(x[i] - centroids[belong],x[i] - centroids[belong])
            J = J / N
            # print ("J = ",J)
            number_of_updates = iteration
            membership = np.array(membership)
            if(abs(J - prevJ) < self.e):
                break
            prevJ = J
            for k in range(K):
                centroids[k, :] = np.sum(x[membership == k,], axis = 0) / np.sum(membership == k)
                # _sum = 0
                # _center = np.zeros((1,D))
                # for i in range(N):
                #     if(assignment[i,k] == 1):
                #         _center = _center + x[i]
                #         _sum = _sum + 1
                # _center = _center / _sum
                # centroids[k] = _center
        return centroids,membership,number_of_updates


        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeans class (filename: kmeans.py)')
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, n_updates = k_means.fit(x)
        centroid_labels = []
        for cluster in range(self.n_cluster):
            unique,counts = np.unique(y[membership == cluster], return_counts=True)
            labelOfCentroid = unique[np.argmax(counts)]
            centroid_labels.append(labelOfCentroid)
        # raise Exception(
        #     'Implement fit function in KMeansClassifier class (filename: kmeans.py)')
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = np.array(centroid_labels)
        self.centroids = centroids

        assert self.centroid_labels.shape == (
        self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        labels = []
        for i in range(N):
            _distance = []
            for k in range(self.n_cluster):
                distance = np.inner(x[i] - self.centroids[k],x[i] - self.centroids[k])
                _distance.append(distance)
            index = np.argmin(np.array(_distance))
            labels.append(self.centroid_labels[index])
        # raise Exception(
        #     'Implement predict function in KMeansClassifier class (filename: kmeans.py)')
        # DONOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)

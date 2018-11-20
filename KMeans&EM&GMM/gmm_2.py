import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape
        # print ("N =",N,",D =",D)

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            k_means = KMeans(n_cluster = self.n_cluster, max_iter = self.max_iter, e = self.e)
            centroids, membership, n_updates = k_means.fit(x)
            variance_k = []
            pi_k = []
            for k in range(self.n_cluster):
                x_k = x[membership == k,:]
                centroid_k = centroids[k]
                covariance = np.zeros((D,D))
                N_k = x_k.shape[0]
                for i in range(N_k):
                    subtract = np.array(x_k[i] - centroid_k).reshape(1,D)
                    covariance = covariance + np.dot(np.transpose(subtract),subtract)
                variance_k.append(covariance/N_k)
                pi_k.append(N_k/N)
            self.pi_k = np.array(pi_k)
            self.variances = np.array(variance_k)
            self.means = np.array(centroids)

            # raise Exception(
            #     'Implement initialization of variances, means, pi_k using k-means')
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.means = np.random.uniform(low = 0.0, high = 1.0, size = (self.n_cluster, D))
            self.variances = np.array([np.identity((D))] * self.n_cluster)
            self.pi_k = np.array([1.0 / self.n_cluster] * self.n_cluster)
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k randomly')
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        # self.means, self.variances, self.pi_k = np.array(centroids), np.array(variance_k), np.array(pi_k)
        # print("self.means =", self.means)
        # print("self.variances =", self.variances)
        # print("self.pi_k =", self.pi_k)
        l = self.compute_log_likelihood(x,self.means,self.variances,self.pi_k)
        Gaussian_pdf = self.Gaussian_pdf
        # print ("l =",l)
        number_of_updates = 0
        for iteration in range(self.max_iter):
            gamma_ik = np.zeros((N, self.n_cluster))
            for i in range(N):
                gamma_ik_temp = []
                for k in range(self.n_cluster):
                    gaussian_pdf = Gaussian_pdf(mean = self.means[k], variance = self.variances[k])
                    gamma_ik_temp.append(self.pi_k[k] * gaussian_pdf.getLikelihood(x[i, :]))
                _sum = np.sum(np.array(gamma_ik_temp))
                gamma_ik_temp = gamma_ik_temp  / _sum
                for k in range(self.n_cluster):
                    gamma_ik[i,k] = gamma_ik_temp[k]

            N_k = np.sum(gamma_ik, axis=0)

            # calculate the centroids
            mu_k = []
            for k in range(self.n_cluster):
                _sum = np.zeros((1, D))
                for i in range(N):
                    _sum += gamma_ik[i,k] * x[i,:]
                _sum = _sum / N_k[k]
                mu_k.append(_sum)
            mu_k = np.array(mu_k).reshape(self.n_cluster,D)

            #calculate the variance
            variance_k = []
            pi_k = []
            for k in range(self.n_cluster):
                _sum = 0
                for i in range(N):
                    _subtract = np.array(x[i,:] - self.means[k,: ]).reshape(1,D)
                    _sum += gamma_ik[i,k] * np.dot(np.transpose(_subtract),_subtract)

                variance_k.append(_sum * 1.0 / N_k[k])
                pi_k.append(N_k[k] * 1.0 / N)
            variance_k = np.array(variance_k)
            pi_k = np.array(pi_k)

            # update all of them
            self.means, self.variances, self.pi_k = np.array(mu_k), np.array(variance_k), np.array(pi_k)
            l_new = self.compute_log_likelihood(x, self.means, self.variances, self.pi_k)
            # print ("l_new = ",l_new)
            if(abs(l_new - l) < self.e):
                break
            number_of_updates = number_of_updates + 1
            l = l_new
        return number_of_updates

        # raise Exception('Implement fit function (filename: gmm.py)')
        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement sample function in gmm.py')
        samples = []
        for i in range(N):
            k = np.argmax(np.random.multinomial(1, self.pi_k, size=1))
            samples.append(np.random.multivariate_normal(self.means[k], self.variances[k]))
        # DONOT MODIFY CODE BELOW THIS LINE
        return np.array(samples)

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        p = []
        N = x.shape[0]
        Gaussian_pdf = self.Gaussian_pdf
        for i in range(N):
            px_i = 0
            for k in range(self.n_cluster):
                # print("means[k] = ", means[k])
                # print("variances[k] = ", variances[k])
                # print("pi_k[",k,"] = ", pi_k[k])
                gaussian_pdf = Gaussian_pdf(mean = self.means[k],variance = self.variances[k])
                px_i = px_i + pi_k[k] * gaussian_pdf.getLikelihood(x[i,:])
                # print("pi_k[",k,"] = ", pi_k[k], ",likehood =",gaussian_pdf.getLikelihood(x[i,:]))
            p.append(px_i)
        log_likelihood = np.sum(np.log(p)).astype(float)
        # return np.float64(loglike).item()
        # raise Exception('Implement compute_log_likelihood function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
        return np.float64(log_likelihood).item()

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            D = self.variance.shape[0]
            variance = self.variance
            while (np.linalg.matrix_rank(variance) < D):
                variance = variance + 0.001 * np.eye(D)
            self.inv = np.linalg.inv(variance)
            self.c = np.power(2 * np.pi,D) * np.linalg.det(variance)
            # raise Exception('Impliment Guassian_pdf __init__')
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            D = self.inv.shape[0]
            xi_mu_k = np.array(x - self.mean).reshape(1,D)
            p = np.exp(-0.5 * (np.dot(np.dot(xi_mu_k, self.inv) , np.transpose(xi_mu_k))))
            # p = np.power(np.linalg.det(2 * np.pi * self.inv), -1 / 2) * p
            p = p / np.sqrt(self.c)
            # print ("p = ",p)
            # raise Exception('Impliment Guassian_pdf getLikelihood')
            # DONOT MODIFY CODE BELOW THIS LINE
            return np.float64(p).item()

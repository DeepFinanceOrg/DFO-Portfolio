


class Cluster_ext:

    def __init__(self):
        pass


    def KmeanClusterSil(feature_matrix,
                        random_state = 1,
                        k_min = 2 ,
                        max_num_clusters = None
                        ):
        silhouette_coefficients = []
        if max_num_clusters != None:
            k_max = max_num_clusters
        else:
            k_max = feature_matrix.shape[0]
        
        for k in range(k_min,k_max):
            kmeans = KMeans(n_clusters = k, init = 'random', random_state= random_state, n_init =50)
            kmeans.fit(feature_matrix)
            score = silhouette_score(feature_matrix, kmeans.labels_)
            silhouette_coefficients.append(score)
        
        optimal_k = np.argmax(silhouette_coefficients) + k_min
        best_kmeans = KMeans(n_clusters= optimal_k ,init = 'random', random_state= random_state, n_init =50)
        best_kmeans.fit(feature_matrix)
        
        clusters = defaultdict(list)
        for cluster_number,asset in zip(best_kmeans.labels_ , feature_matrix.columns):
            clusters[cluster_number].append(asset)
            
        return optimal_k , best_kmeans.labels_.tolist() , clusters
    
    
        # Gap Statistic for K means
    def optimalK_KMeans(data, nrefs=3, maxClusters=9):
        """
        Calculates KMeans optimal K using Gap Statistic 
        Params:
            data: ndarry of shape (n_samples, n_features)
            nrefs: number of sample reference datasets to create
            maxClusters: Maximum number of clusters to test for
        Returns: (gaps, optimalK)
        """
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
        for gap_index, k in enumerate(range(1, maxClusters)):
            # Holder for reference dispersion results
            refDisps = np.zeros(nrefs)
            # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
            for i in range(nrefs):
                
                # Create new random reference set
                randomReference = np.random.random_sample(size=data.shape)
                
                # Fit to it
                km = KMeans(k)
                km.fit(randomReference)
                
                refDisp = km.inertia_
                refDisps[i] = refDisp
            # Fit cluster to original data and create dispersion
            km = KMeans(k)
            km.fit(data)
            
            origDisp = km.inertia_
            # Calculate gap statistic
            gap = np.log(np.mean(refDisps)) - np.log(origDisp)
            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap
            
            resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
        return (gaps.argmax() + 1, resultsdf)

        




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
        

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict



class Cluster_ext:

    def __init__(self):
        pass


    def KmeanClusterSil(feature_matrix:pd.DataFrame,
                        random_state = 1,
                        k_min = 2 ,
                        max_num_clusters = None
                        ):
        """
        KmeanClusterSil:\n
        Clustering with Kmeans method and finding the optimal number of 
        clusters based on Silhouette score.
        ---------------------------
        Parameters:
            - feature_matrix: pandas.DataFrame
            - random_state: int, default is 1.
            - k_min: int, default is 2.
            - max_num_clusters: int, default is None.
        ---------------------------
        Return:
            - optimal_k: int
            - best_kmeans.labels_.tolist: list
            - clusters: dict
        ---------------------------
        """
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
    
    
    def KmeanClusterGap(feature_matrix:pd.DataFrame,
                        nrefs=3,
                        max_num_clusters=None ,
                        random_state = 1
                        ):
        
        """
        KmeanClusterGap:\n
        Clustering with Kmeans method and finding the optimal number of 
        clusters based on Gap-statistic.
        ---------------------------
        Parameters:
            - feature_matrix: pandas.DataFrame
            - nrefs: int, default is 3.
            - max_num_clusters: int, default is None.
            - random_state: int, default is 1.
        ---------------------------
        Return:
            - optimal_k: int
            - Resultsdf: pandas.DataFrame
            - clusters: dict
        ---------------------------
            
        """
        if max_num_clusters == None:
            maxClusters = feature_matrix.shape[0]
        else:
            maxClusters = max_num_clusters
            
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
        for gap_index, k in enumerate(range(1, maxClusters)):
            # Holder for reference dispersion results
            refDisps = np.zeros(nrefs)
            # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
            for i in range(nrefs):
                
                # Create new random reference set
                randomReference = np.random.random_sample(size=feature_matrix.shape)
                
                # Fit to it
                km = KMeans(k)
                km.fit(randomReference)
                
                refDisp = km.inertia_
                refDisps[i] = refDisp
            # Fit cluster to original feature_matrix and create dispersion
            km = KMeans(k)
            km.fit(feature_matrix)
            
            origDisp = km.inertia_
            # Calculate gap statistic
            gap = np.log(np.mean(refDisps)) - np.log(origDisp)
            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap
            resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
            optimal_k = gaps.argmax() + 1
            best_kmeans = KMeans(n_clusters= optimal_k ,init = 'random',
                                random_state= random_state)
            best_kmeans.fit(feature_matrix)
            
            clusters = defaultdict(list)
            for cluster_number,asset in zip(best_kmeans.labels_ , feature_matrix.columns):
                clusters[cluster_number].append(asset)
        return optimal_k, resultsdf , clusters



    def getQuasiDiag(link:np.array): 
        
        """
        getQuasiDiag:\n
        Gets Quasi-Diagonaliztion (actually Matrix Seriation) of the assets
        ---------------------------
        Parameters:
            - link: numpy.array
        ---------------------------
        Return:
            - sortIx.tolist(): list
        ---------------------------
            
        """ 
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3] 
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2) 
            df0 = sortIx[sortIx >= numItems]
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = sortIx.append(df0)
            sortIx = sortIx.sort_index()  
            sortIx.index = range(sortIx.shape[0])
        return sortIx.tolist() 
            

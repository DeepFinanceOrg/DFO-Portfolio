import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict



class Cluster_ext:

    def __init__(self):
        pass


    def KmeanClusterSil(feature_matrix:pd.DataFrame,
                        random_state:int = 1,
                        k_min:int = 2 ,
                        max_num_clusters:int = None
                        ) -> dict:
        """
        KmeanClusterSil:\n
        Clustering with Kmeans method and finding the optimal number of 
        clusters based on Silhouette score.
        ---------------------------
        Parameters:
            - feature_matrix: pandas.DataFrame
            - random_state: int, default is 1.
            - k_min: int, default is 2.
            - max_num_clusters: int, Maximum number of clusters to test for and default is None.
        ---------------------------
        Return:
            - clusters: dict, A Dictionary that contains assets in each cluster seperately.
        ---------------------------
        """
        silhouette_coefficients = []
        if max_num_clusters != None:
            k_max = max_num_clusters
        else:
            k_max = feature_matrix.shape[0]
        
        for k in range(k_min,k_max):
            kmeans = KMeans(n_clusters = k,
                            init = 'random',
                            random_state= random_state,
                            n_init =50).fit(feature_matrix)
            silhouette_coefficients.append(silhouette_score(feature_matrix, kmeans.labels_))
        
        best_kmeans = KMeans(n_clusters= np.argmax(silhouette_coefficients) + k_min,
                             init = 'random',
                             random_state= random_state,
                             n_init =50).fit(feature_matrix)
        
        clusters = defaultdict(list)
        for cluster_number,asset in zip(best_kmeans.labels_ , feature_matrix.columns):
            clusters[cluster_number].append(asset)
            
        return clusters
    
    
    def KmeanClusterGap(feature_matrix:pd.DataFrame,
                    data_rets:pd.DataFrame,
                    nrefs:int=20,
                    max_num_clusters:int=None ,
                    random_state:int= 11,
                    
                    ):
        """
        KmeanClusterGap:\n
        Clustering with Kmeans method and finding the optimal number of 
        clusters based on Gap-statistic.
        ---------------------------
        Parameters:
            - feature_matrix: pandas.DataFrame
            - data_rets: pandas.DataFrame , dataframe of returns of assets
            - nrefs: int, number of sample reference datasets to create and default is 3.
            - max_num_clusters: int, Maximum number of clusters to test for and default is None.
            - random_state: int, default is 1.
        ---------------------------
        Return:
            - clusters: dict, A Dictionary that contains assets in each cluster seperately.
        ---------------------------
            
        """
        
        if max_num_clusters == None:
            maxClusters = feature_matrix.shape[0]
        else:
            maxClusters = max_num_clusters
            
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
        for gap_index, k in enumerate(range(1, maxClusters)):
            refDisps = np.zeros(nrefs)
            for i in range(nrefs):
                
                # randomReference_rets = (pd.DataFrame(np.random.rand(*data_rets.shape)))
                # randomReference_corrs = (np.array((pd.DataFrame(np.random.rand(*data_rets.shape))).corr()))
                randomReference = np.sqrt(2 * (1 - (np.array((pd.DataFrame(np.random.rand(*data_rets.shape))).corr()))).round(5))
                
                km = KMeans(k).fit(randomReference)
                
                refDisps[i] = km.inertia_
            km = KMeans(k).fit(feature_matrix)
            
            gap = np.mean(np.log(refDisps)) - np.log(km.inertia_)
            gaps[gap_index] = gap
            resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap},
                                         ignore_index=True)
            best_kmeans = KMeans(n_clusters= gaps.argmax() + 1,
                                 init = 'random',
                                random_state= random_state).fit(feature_matrix)
            
            clusters = defaultdict(list)
            for cluster_number,asset in zip(best_kmeans.labels_ , feature_matrix.columns):
                clusters[cluster_number].append(asset)
        return clusters



    def getQuasiDiag(link:np.array): 
        
        """
        getQuasiDiag:\n
        Gets Quasi-Diagonaliztion (actually Matrix Seriation) of the assets
        ---------------------------
        Parameters:
            - link: numpy.array, Linkage matrix.
        ---------------------------
        Return:
            - sortIx.tolist(): list, List of sorted asset indeces based on their simillarity.
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
            

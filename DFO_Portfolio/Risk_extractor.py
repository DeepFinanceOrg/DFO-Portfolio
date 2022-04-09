import numpy as np , pandas as pd 


#TODO: Complete functions notes

class Covariance_base:

    def __init__(
        self,
        *,Risk_extractor_Model='Risk_Parity',
        Inside_cluster = False
    )-> None:
        
        """
        Parameters
        ----------
        * : TYPE
            DESCRIPTION.
        Risk_extractor_Model : TYPE, optional
            DESCRIPTION. The default is 'Risk_Parity'.
        Inside_cluster : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None
            DESCRIPTION.

        """
        self.__Risk_extractor = Risk_extractor_Model
        self.__Inside_cluster = Inside_cluster

    @staticmethod
    def Risk_Parity(cov)-> np.array:
        """
        

        Parameters
        ----------
        cov : TYPE
            DESCRIPTION.

        Returns
        -------
        ivp : TYPE
            DESCRIPTION.

        """
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    @staticmethod 
    def Optimal_Portfolio(cov,*,mu=None)-> np.array:
        """
        Parameters
        ----------
        cov : TYPE
            DESCRIPTION.
        * : TYPE
            DESCRIPTION.
        mu : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        w : TYPE
            DESCRIPTION.

        """
        if mu is None : mu = np.ones(shape=(cov.shape[0],1))
        w = np.dot(np.linalg.inv(cov),mu)
        w /= np.dot(mu.T,w)
        return w 

    def Get_Cluster_RParity(
        self,*,
        cov,
        cluster_items
    )-> float:
        
        """
        Parameters
        ----------
        * : TYPE
            DESCRIPTION.
        cov : TYPE
            DESCRIPTION.
        cluster_items : TYPE
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        """
        w_ = self.Risk_Parity(
            cov.iloc[cluster_items,cluster_items]
        ).reshape(-1,1)
        
        return np.dot(
            np.dot(w_.T,cov.iloc[cluster_items,cluster_items]),w_
        )[0,0]
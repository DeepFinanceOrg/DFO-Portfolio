import  numpy as np 
import pandas as pd 


class Covariance_base:
    
    def __init__(self,
                 *,
                 Risk_extractor_Model='Risk_Parity',
                 Inside_cluster = False
                 ) -> None:    
        """
        Returns
        -------
        None
            DESCRIPTION.
                
           TODO: Complete the notes  
        """
        
        self.__Risk_extractor = Risk_extractor_Model
        self.__Inside_cluster = Inside_cluster
    
    
    @staticmethod
    def Risk_Parity(cov) -> np.array:
        
        """
        Parameters
        ----------
        cov : TYPE
            DESCRIPTION.

        Returns
        -------
        Risk_Parity : TYPE
                    DESCRIPTION.

           TODO: Complete the notes
        """
        
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        
        return ivp
    
    def Get_Cluster_RParity(self,
                              cov : pd.DataFrame,
                              cluster_items : pd.DataFrame.iloc) -> int:
        """
        Parameters
        ----------
        cov : TYPE
            DESCRIPTION.
            ...
        Returns
        -------
        int
            DESCRIPTION.
            
           TODO: Complete the notes            
        """
        w_ = self.Risk_Parity(
            cov.iloc[
                cluster_items,cluster_items
                ]
            ).reshape(-1,1)
        
        return np.dot(
            np.dot(w_.T,cov.iloc[cluster_items,cluster_items]),w_
            )[0,0]
        
    @staticmethod
    def Optimal_Portfolio(cov,*,mu=None) -> np.array:
        """
        Parameters
        ----------
        cov : TYPE
            DESCRIPTION.
        mu : TYPE
            DESCRIPTION.
        Returns
        -------
        None.

        TODO : Complete the notes  
        """
        if mu is None : mu = np.ones(shape=(cov.shape[0],1))
        w = np.dot(np.linalg.inv(cov),mu)
        w /= np.dot(mu.T,w)
        return w 
        
    
    

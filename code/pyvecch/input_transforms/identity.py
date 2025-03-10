"""identity.py
Identity input transform. 
Author: Felix Jimenez
"""
import torch
from .input_transform import InputTransform

class Identity(InputTransform):
    """Identity transform..
    
    No transform is performed, this class allows a simpler 
    interface for other code.

    """
    def __init__(self, d):
        super(Identity, self).__init__(d)
        
    def transform_query(self, x):
        """ Return the original inputs.. 
        
        Return the inputs on their original scale.

        Args:
            x (torch.Tensor): Inputs. 

        Returns:
            x_tilde (torch.Tensor) Original inputs. 
        """
        x_tilde = x
        return x_tilde

    def transform_covar(self, x):
        """ Return the original inputs.. 
        
        Return the inputs on their original scale.

        Args:
            x (torch.Tensor): Inputs. 

        Returns:
            x_tilde (torch.Tensor) Original inputs. 
        """
        x_tilde = x
        return x_tilde

    def update_transform(self, model):
        """No update needed. 
        Args:
            model (VecchiaGP): GP model for consistency. 
        """
        pass
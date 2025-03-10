from .input_transform import InputTransform
import torch
from .base_warp import BaseWarp

class Warping(InputTransform):
    """Warped input scaling transform. 
    
    The input transform for non-linear warping as described
    in https://arxiv.org/pdf/2203.01459.pdf. Each input dimension 
    is mapped through a non-linear bijective function. 
    """
    def __init__(self, d, **kwargs):
        """Warped input scaling transform.
        Args:
            d (int): Input dimension.
        """
        super(Warping, self).__init__(d)
        indices = [i for i in range(d)]#torch.arange(d)
        self.base_warp = BaseWarp(indices, **kwargs)
        self.scales = torch.ones(d)

    def transform_query(self, x):
        """ Warp and scale inputs.
        
        Use Kumaraswamy to warp inputs and then linearly scale them.

        Args:
            x (torch.Tensor): Inputs to warp. 

        Returns:
            x_tilde (torch.Tensor) Warped and scaled inputs. 
        """
        x_tilde = self.base_warp._transform(x)
        x_tilde = torch.div(x_tilde, self.scales)
        return x_tilde

    def transform_covar(self, x):
        """ Warp inputs.
        
        Use Kumaraswamy to warp inputs.

        Args:
            x (torch.Tensor): Inputs. 

        Returns:
            x_tilde (torch.Tensor) Warped inputs. 
        """
        x_tilde = self.base_warp._transform(x)
        return x_tilde


    def update_transform(self, model):
        """Update the linear scaling to use new lengthscales. 

        Given a VecchiaGP model, use the lengthscales from its covariance 
        module and use these values for linear scaling. 

        Args:
            model (VecchiaGP): GP model whose lengthscales we want to use. 
        """
        tkwargs = model.tkwargs
        self.scales = model.covar_module.base_kernel.lengthscale.data.to(**tkwargs)
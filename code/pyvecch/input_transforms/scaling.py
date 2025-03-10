import torch
from .input_transform import InputTransform

class Scaling(InputTransform):
    """Linear scaling input transform.
    
    The input transform for scaled Vecchia https://arxiv.org/abs/2005.00386. 
    Given an ARD kernel with d lengthscales values the input dimensions are 
    divided by the corresponding lengthscales. 

    """
    def __init__(self, d):
        """Linear scaling input transform.
        Args:
            d (int): Dimension of the inputs. 
        """
        super(Scaling, self).__init__(d)
        self.scales = torch.ones(d)

    def transform_query(self, x):
        """ Linearly scale the inputs using lengthscales. 
        
        Scale each of the dimensions of x by the inverse lengthscales
        of a VecchiaGP. 

        Args:
            x (torch.Tensor): Inputs to scale. 

        Returns:
            x_tilde (torch.Tensor) Linearly scaled inputs. 
        """
        x_tilde = torch.div(x, self.scales)
        return x_tilde

    def transform_covar(self, x):
        """ Return the original inputs.. 
        
        Return the inputs on their original scale for
        use with an ARD kernel.

        Args:
            x (torch.Tensor): Inputs. 

        Returns:
            x_tilde (torch.Tensor) Original inputs. 
        """
        x_tilde = x
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


    

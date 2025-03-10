import torch
from gpytorch.utils.grid import ScaleToBounds

class InputTransform(torch.nn.Module):
    """ Base class for all input transforms. 
    Inputs transforms are objects that store the information required to map inputs
    to a transformed space. For example, the lengthscales of a GP in the scaled Vecchia
    transform. 
    """
    def __init__(self, d):
        super(InputTransform, self).__init__()
        self.scale_to_bounds = ScaleToBounds(-1,1)
        self.d = d
        
    def transform_query(self, x):
        """Transform the inputs for query.
        Args:
            x (torch.Tensor): Inputs to transform. 
        Returns:
            x_tilde (torch.Tensor): Inputs that have been transformed.
        """
        raise NotImplementedError

    def transform_covar(self, x):
        """Transform the inputs for computing covariance
        Args:
            x (torch.Tensor): Inputs to transform. 
        Returns:
            x_tilde (torch.Tensor): Inputs that have been transformed.
        """
        raise NotImplementedError

    def update_transform(self, model):
        """Alter the transform based on Vecchia GP. 
        Args:
            model (VecchiaGP): GP model to use to update the input transform. 
        """
        raise NotImplementedError


    def forward(self, x):
        return self.transform_covar(x)
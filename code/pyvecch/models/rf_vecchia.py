from .vecchia_gp import VecchiaGP
import torch
from ..input_transforms import Identity

class RFVecchia(VecchiaGP):
    """Response first Vecchia GP. 
    
    A Vecchia GP where the observed data is assumed to be ordered
    before all other data. 
    """

    def __init__(self, covar_module, mean_module, likelihood, 
    neighbor_oracle, prediction_strategy, input_transform=Identity):
        """Response first Vecchia GP. 
        Args:
            covar_module (gpytorch.kernels.Kernel): Prior covariance function for the GP. 
            mean_module (gpytorch.means.Mean): The prior mean for the GP. 
            likelihood (gpytorch.likelihoods.Likelihood): Likelihood for the data. 
            neighbor_oracle (NeighborOracle): A class extending the NeighborOracle class
                and used to compute conditioning sets and order the data. 
            prediction_strategy (PredictionStrategy): A class extending the PredictionStrategy
                class. This is used to make predicitions at unseen locations. For example 
                response first predictions. 
            input_transform (InputTransform): A class extending InputTransform base class.
                defaults to pyvecch.input_transforms.Identity.
        """
        super(RFVecchia, self).__init__(
            covar_module, mean_module,likelihood, 
            neighbor_oracle, prediction_strategy, input_transform)

    def forward(self, subset):
        """Get the latent distribution of the subset of the data. 
        Args:
            subset (np.array): Indices of points in the likelihood to evaluate. 
        Returns:
            cond_dist (torch.distributions.MultivariateNormal): Conditional distribution 
                    for the subset given its conditioning set. 
            """
        g = self.neighbor_oracle.get_neighbors(subset)
        xg, yg = self.neighbor_oracle[g]
        yg = yg.unsqueeze(-1)
        X, Y = self.neighbor_oracle[subset]
        X = X.unsqueeze(1)
        cond_dist = self.conditional_distribution(X, Y, xg, yg, g)
        return cond_dist


    def conditional_distribution(self, X, Y, xg, yg, g):
        """Get conditional distribution on f(X) given yg.

        Get the conditional distribution of Y(X) given yg(xg) using the 
        Vecchia approximation. 

        Args:
            X (torch.Tensor): Inputs at which we want distribution on f(X).
            Y (torch.Tensor): Observed values at X. 
            xg (torch.Tensor): Inputs of the conditioning set for observations Y. 
            yg (torch.Tensor): Observed values at xg. 
        Returns:
            cond_dist (torch.distributions.MultivariateNormal): Conditional distribution 
                for f(X) given yg. 
        """
        # create a mask to deal with nonsquare batches of conditioning sets. 
        x_mask = (1.0*(g>=0)).unsqueeze(-1).to(**self.tkwargs)
        mask = x_mask @ x_mask.transpose(-1,-2)
        # computing (k_{x_g, x_g}+I\sigma^2)^{-1}, but with a mask to deal
        # with nonsquared conditioning sets. 
        k_temp = self.compute_covar(xg).add_diag(self.likelihood.noise).add_jitter().evaluate()
        k_temp_L = torch.linalg.cholesky(k_temp)
        masked_inv = torch.linalg.solve_triangular(
            k_temp_L, 
            torch.eye(k_temp_L.shape[-1]).unsqueeze(0).to(**self.tkwargs), 
            upper = False) * mask

        masked_inv = masked_inv.transpose(-1,-2) @ masked_inv
        cached_term = self.compute_covar(X, xg).evaluate() @ masked_inv
        cond_cov = self.compute_covar(X).evaluate() -  cached_term @ self.compute_covar(xg,X).evaluate()

        cond_mean = (cached_term @ yg).squeeze()
        cond_cov = torch.clip(cond_cov.squeeze(), min = 1e-9).sqrt()
        covar = torch.diag_embed(cond_cov)
        cond_dist = torch.distributions.MultivariateNormal(cond_mean, covar)
        return cond_dist
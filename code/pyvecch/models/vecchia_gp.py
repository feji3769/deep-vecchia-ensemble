from gpytorch.models import GP
from gpytorch.distributions import MultivariateNormal as mvn
from ..input_transforms.identity import Identity

class VecchiaGP(GP):
    """ Base class for all VecchiaGPs. 
    VecchiaGPs contain three components (1) a gpytorch.likelihoods.Likelihood, 
    (2) a class extending a NeighborOracle and a (3) class extending a 
     PredictionStrategy.

    Attributes:
        tkwargs (dict): type kwargs of the model. Includes device and datatype. 
    """
    def __init__(self, 
        covar_module, mean_module, likelihood, neighbor_oracle, 
        prediction_stategy, input_transform=Identity):
        """Base class for all VecchiaGPs. """
        super(VecchiaGP, self).__init__()
        self.likelihood = likelihood
        self.neighbor_oracle = neighbor_oracle
        self.prediction_stategy = prediction_stategy
        self.tkwargs = {"dtype":self.neighbor_oracle.x.dtype, 'device':self.neighbor_oracle.x.device}
        self.mean_module = mean_module.to(**self.tkwargs)
        self.covar_module = covar_module.to(**self.tkwargs)
        self.input_transform = input_transform

    def sort_data(self, inputs, targets, **kwargs):
        '''Sort the inputs and targets using the sorting strategy.
            
        Args:
            inputs (torch.Tensor): Input data to sort by. 
            targets (torch.Tensor): Output data corresponding to inputs. 
        Returns:
            Sorted versions of the inputs and targets. 
        '''
        ordered_indices = self.sorting_strategy(inputs, **kwargs)
        sorted_inputs, sorted_targets = inputs[ordered_indices], targets[ordered_indices]
        return sorted_inputs, sorted_targets

    def compare_neighbors(self, x, xn, dn):
        """Compare neighbor set of x to x. 
        Compare the distances between the elements of x, 
        and its neighbors xn (distances given by dn). Return the xn made 
        of the elements of x and xn with the smallest distances. 

        Args:
            x (torch.Tensor): (batch_size x n x d) ordered query points. 
            xn (torch.Tensor): (batch_size x n x m x d) neighbors of query points.
            dn (torch.Tensor): (batch_size x n x d) distance between query points and nbrs. 
        Returns:
            xn_new (torch.Tensor): (batch_size x n x m x d) neighbor set closest to x. 
        """
        # TODO: compute correct xn for x, and provide what's needed by JointRF.
        pass

    def query_neighbors(self, x_query, k, **kwargs):
        '''Get the nearest neighbors of x_query in the existing database.

        Args:
            x_query (torch.Tensor):  batchsize x d.
            k (int): number of neighbors to return.

        Returns:
            (torch.Tensor) batchsize x d x k.
        '''
        x_query_transformed = self.input_transform.transform_query(x_query)
        return self.neighbor_oracle.query(x_query_transformed, k, **kwargs)

    def compute_covar(self, x1, x2=None):
        '''Compute the covariance between x1 and x2.
        Trasnform x1 and x2 using input transform, 
        and then compute the covariance between them. 

        Args:
            x_query (torch.Tensor):  batchsize x d.
            k (int): number of neighbors to return.

        Returns:
            (torch.Tensor) batchsize x d x k.
        ''' 
        x1_transformed = self.input_transform.forward(x1)
        if x2 is not None:
            x2_transformed = self.input_transform.forward(x2)
            return self.covar_module(x1_transformed, x2_transformed)
        else:
            return self.covar_module(x1_transformed)
        

    def update_transform(self):
        """Update the transform using the current Vecchia GP model."""
        self.input_transform.update_transform(self)
        self.neighbor_oracle.reorder_data(self.input_transform)

    @property
    def m(self):
        """int: Number of nearest neighbors."""
        return self.neighbor_oracle.m

    @property
    def n(self):
        """int: Number of observations in the training set."""
        return self.neighbor_oracle.x.shape[0]
        
    def __call__(self, inputs, **kwargs):
        # if the model is in training mode the inputs should be 
        # indices of data to include in the mini-batch.
        if self.training:
            dist =  super().__call__(inputs, **kwargs)
        else:
        # if the model is in eval model, then the inptus will be 
        # test locations and we get a posterior using the prediction
        # strategy.
            dist = self.prediction_stategy(inputs, self, **kwargs)
        
        # extract the mean and covariance from the output. 
        mu = dist.mean
        covar = dist.covariance_matrix
        return mvn(mu, covar)

    def posterior(self, inputs, **kwargs):
        dist = self.prediction_stategy(inputs, self, **kwargs)
        mu = dist.mean
        covar = dist.covariance_matrix
        return mvn(mu, covar)
from torch.distributions import MultivariateNormal as MVN


class VecchiaPrediction():
    def get_U(self, x, model, conditioning_set, **kwargs):
        '''should return U_lp and U_pp_inv.'''
        raise NotImplementedError

    def get_conditioning_set(self, x, model, **kwargs):
        ''' at a minimum this should return a dictionary with one key 
        "conditioning_indices" which maps to (torch.Tensor) indices of conditioning points.'''
        raise NotImplementedError 


    def posterior(self, x, model, **kwargs):
        '''get the posterior predictive distribution for a general Vecchia GP.'''
        conditioning_set = self.get_conditioning_set(x, model, **kwargs)
        U_lp, U_pp_inv = self.get_U(x, model, conditioning_set, **kwargs)
        z_o = model.neighbor_oracle.y[conditioning_set['conditioning_indices']].unsqueeze(-1)
        mu_hat = U_lp @ z_o
        mu_hat = mu_hat.squeeze(-1)
        mu_hat =  -1 * U_pp_inv @ (mu_hat)
        mu_hat = mu_hat.squeeze(-1)
        sigma_hat = U_pp_inv @ U_pp_inv.transpose(-1,-2)

        pred = MVN(mu_hat, sigma_hat)


        return pred

    def __call__(self, test_x, model, **kwargs):
        if len(test_x.shape) < 3:
            x_p = test_x.unsqueeze(0)
        else:
            x_p = test_x
        return self.posterior(x_p, model, **kwargs)


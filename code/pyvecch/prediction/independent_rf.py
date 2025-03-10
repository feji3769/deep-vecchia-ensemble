import torch
from .vecchia_prediction import VecchiaPrediction

class IndependentRF(VecchiaPrediction):
    def get_conditioning_set(self, test_x, model, **kwargs):
        '''
        test_x (shape = (b,q,d)): torch.tensor of test locations. 
        '''
        # query conditioning set. 
        nn_dist_batch, nn_ind_batch = model.query_neighbors(
            test_x, 
            model.m, 
            is_self = kwargs.get("is_self", False)
            )
        xq_g = model.neighbor_oracle.x[nn_ind_batch, :]

        output = {
            "conditioning_indices":nn_ind_batch, 
            "conditioning_distances":nn_dist_batch, 
            "conditioning_inputs":xq_g
        }
        return output

    def get_U(self, test_x, model, conditioning_set, **kwargs):  
        tkwargs = model.tkwargs
        xq_g = conditioning_set['conditioning_inputs']
        b_t, ds = self.get_b_d(model, xq_g, test_x)
        U_lp = (-1 * b_t * 1/ds.sqrt())
        U_pp = 1/(ds.squeeze(-1).squeeze(-1).clip(1e-9).sqrt())
        U_pp = torch.diag_embed(U_pp)
        U_pp_inv = torch.linalg.solve_triangular(U_pp,torch.eye(U_pp.shape[-1], **tkwargs), upper = True)
        return U_lp, U_pp_inv


    def get_b_d(self, model, x_g, test_x):
        def k(x1,x2=None):
            return model.compute_covar(x1,x2)
        Kgg = k(x_g).add_diag(model.likelihood.noise); 
        Kgx = k(x_g, test_x.unsqueeze(-2)).evaluate()
        b_t = Kgg.inv_matmul(Kgx).transpose(-2,-1)
        ds = (k(test_x.unsqueeze(-2)) - b_t @ Kgx).evaluate()
        return b_t, ds
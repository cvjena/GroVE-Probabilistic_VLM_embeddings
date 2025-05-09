import gpytorch
import torch

class LVM(gpytorch.models.ApproximateGP):
    def __init__(self, data_dim, latent_dim, n_inducing):

        self.batch_shape = torch.Size([data_dim])

        self.inducing_points = torch.randn(data_dim, n_inducing, latent_dim)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            self.inducing_points.size(-2), batch_shape=self.batch_shape)
        

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.VariationalStrategy(self, self.inducing_points, variational_distribution, learn_inducing_locations=True)

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            variational_strategy,
            num_tasks=data_dim,
            num_latents=data_dim,
            latent_dim=-1
        )


        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=self.batch_shape)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=self.batch_shape),
            batch_shape=self.batch_shape)      
        
        
    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)
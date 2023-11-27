"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct as D
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
import matplotlib.pyplot as plt

# global variables
DOMAIN = np.array([[0, 10]])  # restrict theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
PRIOR_MEAN = 4  # prior mean of v
OPT_RESTARTS = 40  # number of restarts for the optimization
SHOW_PLOTS = False

F_GPR_PARAMS = {
    "kernel": C(0.5, constant_value_bounds="fixed")
    * Matern(nu=2.5, length_scale=1.0, length_scale_bounds=[0.2, 20])
    + WhiteKernel(noise_level=0.15**2),
    "alpha": 1e-10,
    "optimizer": "fmin_l_bfgs_b",
    "n_restarts_optimizer": 5,
    "normalize_y": False,
    "copy_X_train": True,
    "n_targets": None,
    "random_state": None,
}

V_GPR_PARAMS = {
    "kernel": C(1, constant_value_bounds=[0.5, 1.5])
    * D(sigma_0=0, sigma_0_bounds="fixed")
    + C(np.sqrt(2), constant_value_bounds="fixed")
    * RBF(length_scale=1.0, length_scale_bounds=[0.2, 20])
    + WhiteKernel(noise_level=0.0001**2),
    "alpha": 1e-10,
    "optimizer": "fmin_l_bfgs_b",
    "n_restarts_optimizer": 5,
    "normalize_y": False,
    "copy_X_train": True,
    "n_targets": None,
    "random_state": None,
}


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo:
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.f = GaussianProcessRegressor(**F_GPR_PARAMS)
        self.v = GaussianProcessRegressor(**V_GPR_PARAMS)

        # for acquisition function with exp(v-SAFETY_THRESHOLD) use l = 1.0
        # for acquisition function with v^3 use l = 0.034
        self.l = 0.034
        
        self.f_coeff = 2
        self.v_coeff = 2

        self.f_data = np.array([])
        self.v_data = np.array([])
        self.x_data = np.array([]).reshape(-1, 1)

        self.random_step_clock = 1
        self.random_step_period = np.inf

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        if self.random_step_clock == self.random_step_period:
            self.random_step_clock = 1
            return np.random.uniform(*DOMAIN[0])

        self.random_step_clock += 1

        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization OPT_RESTARTS times and pick best solution
        for _ in range(OPT_RESTARTS):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * np.random.rand(
                DOMAIN.shape[0]
            )
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN, approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)

        # TODO: Implement the acquisition function you want to optimize.

        y_mean, y_std = self.f.predict(x, return_std=True)
        v_mean, v_std = self.v.predict(x, return_std=True)

        f = y_mean + self.f_coeff * y_std
        v = (v_mean + PRIOR_MEAN) + self.v_coeff * v_std

        # use either self.l * v**3 or self.l * np.exp(v - SAFETY_THRESHOLD)
        val = f - self.l * v**3
        
        return val

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.

        self.f_data = np.append(self.f_data, f)
        self.v_data = np.append(self.v_data, v)
        self.x_data = np.vstack([self.x_data, [x]])

        self.f.fit(self.x_data, self.f_data)
        self.v.fit(self.x_data, self.v_data - PRIOR_MEAN)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        v_mask = self.v_data < SAFETY_THRESHOLD
        fs = self.f_data[v_mask]
        xs = self.x_data[v_mask]
        best_ind = np.argmax(fs)
        x_max = xs[best_ind]
        return x_max

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        x = np.linspace(*DOMAIN[0], 1000).reshape(-1, 1)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot for f
        f_true = np.vectorize(f)(x)
        ax1.plot(x, f_true, label="f(x) true", linestyle="--", color="orange")
        f_mean, f_std = self.f.predict(x, return_std=True)
        ax1.plot(x, f_mean, label="f(x) surrogate", color="blue")
        ax1.fill_between(
            x[:, 0],
            f_mean - 1.96 * f_std,
            f_mean + 1.96 * f_std,
            alpha=0.2,
            color="blue",
            label="95% confidence interval",
        )
        ax1.set_title("Function f(x)")
        ax1.legend()

        # Plot for v
        v_true = np.vectorize(v)(x)
        ax2.plot(x, v_true, label="v(x) true", linestyle="--", color="orange")
        v_mean, v_std = self.v.predict(x, return_std=True)
        v_mean += PRIOR_MEAN
        ax2.plot(x, v_mean, label="v(x) surrogate", color="blue")
        ax2.fill_between(
            x[:, 0],
            v_mean - 1.96 * v_std,
            v_mean + 1.96 * v_std,
            alpha=0.2,
            color="blue",
            label="95% confidence interval",
        )
        ax2.set_title("Function v(x)")
        ax2.legend()

        # Plot recommendation if required
        if plot_recommendation:
            recommendation = self.next_recommendation()
            ax1.scatter(recommendation, f(recommendation), label="Next recommendation")
            ax2.scatter(recommendation, v(recommendation), label="Next recommendation")

        return fig, (ax1, ax2)


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---


def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return -np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    if SHOW_PLOTS:
        agent.plot()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point([x_init], obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        if SHOW_PLOTS:
            agent.plot()

        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), (
        f"The function get solution must return a point within the"
        f"DOMAIN, {solution} returned instead"
    )

    # Compute regret
    regret = 0 - f(solution)

    print(
        f"Optimal value: 0\nProposed solution {solution}\nSolution value "
        f"{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n"
    )


if __name__ == "__main__":
    main()

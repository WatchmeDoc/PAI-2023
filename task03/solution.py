"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct as D
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
import matplotlib.pyplot as plt

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
PRIOR_MEAN = 4  # prior mean of v
OPT_RESTARTS = 20  # number of restarts for the optimization

F_GPR_PARAMS = {
    "kernel": 0.5 * Matern(1, nu=2.5) + WhiteKernel(noise_level=0.15**2),
    "alpha": 1e-10,
    "optimizer": "fmin_l_bfgs_b",
    "n_restarts_optimizer": 5,
    "normalize_y": False,
    "copy_X_train": True,
    "n_targets": None,
    "random_state": None,
}

V_GPR_PARAMS = {
    "kernel": D(sigma_0=2) + np.sqrt(2) * RBF(1.0) + WhiteKernel(noise_level=0.0001**2),
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
        self.l = 0.1
        self.v_coeff = 1.96

        self.f_data = []
        self.v_data = []
        self.x_data = []

        self.debug = []

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

        return np.array([self.optimize_acquisition_function()])

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

        # Restarts the optimization 20 times and pick best solution
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
        f = y_mean + 1.96 * y_std  # 95% confidence interval
        v = v_mean + self.v_coeff * v_std
        self.debug.append((v_mean, v_std))
        return f - self.l * max(v, 0)

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

        self.f_data.append(f)
        self.v_data.append(v - PRIOR_MEAN)
        self.x_data.append(x)
        self.f.fit(np.stack(self.x_data), np.array(self.f_data))
        self.v.fit(np.stack(self.x_data), np.array(self.v_data))

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        x_max = None
        f_max = -np.inf
        for i in range(len(self.x_data)):
            if self.v_data[i] + PRIOR_MEAN < SAFETY_THRESHOLD and self.f_data[i] > f_max:
                x_max = self.x_data[i]
                f_max = self.f_data[i]
        return x_max

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        x = np.linspace(*DOMAIN[0], 1000)
        plt.figure()
        plt.title("f(x) and v(x)")
        f_mean, f_std = self.f.predict(x, return_std=True)
        plt.plot(x, f_mean, label="f(x)")
        plt.fill_between(
            x,
            f_mean - 1.96 * f_std,
            f_mean + 1.96 * f_std,
            alpha=0.2,
            label="95% confidence interval",
        )
        v_mean, v_std = self.v.predict(x, return_std=True)
        plt.plot(x, v_mean, label="v(x)")
        plt.fill_between(
            x,
            v_mean - 1.96 * v_std,
            v_mean + 1.96 * v_std,
            alpha=0.2,
            label="95% confidence interval",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        if plot_recommendation:
            plt.scatter(self.next_recommendation(), 0, label="Next recommendation")
        plt.legend()
        plt.show()


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

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
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

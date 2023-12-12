import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class NeuralNetwork(nn.Module):
    """
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        hidden_layers: int,
        activation: str,
    ):
        super(NeuralNetwork, self).__init__()

        # change: Implement this function which should define a neural network
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self._layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers):
            self._layers.append(nn.Linear(hidden_size, hidden_size))
            self._layers.append(nn.ReLU())
        self._layers.append(nn.Linear(hidden_size, output_dim))
        self.model = nn.Sequential(self._layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # change: Implement the forward pass for the neural network you have defined.
        return self.model(s)


class Actor:
    def __init__(
        self,
        hidden_size: int,
        hidden_layers: int,
        actor_lr: float,
        state_dim: int = 3,
        action_dim: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.model = None
        self.optimizer = None
        self.setup_actor()

    def setup_actor(self):
        """
        This function sets up the actor network in the Actor class.
        """
        # change: Implement this function which sets up the actor network.
        if self.model is None:
            self.model = NeuralNetwork(
                input_dim=self.state_dim,
                output_dim=self.action_dim,
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_layers,
                activation="relu",
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        """
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        """
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(
        self, state: torch.Tensor, deterministic: bool
    ) -> (torch.Tensor, torch.Tensor):
        """
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        """
        assert (
            state.shape == (3,) or state.shape[1] == self.state_dim
        ), "State passed to this method has a wrong shape"
        # TODO: sample using mu and sigma if not deterministic else mu. log_prob ??
        action, log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped
        # using the clamp_log_std function.
        assert action.shape == (state.shape[0], self.action_dim) and log_prob.shape == (
            state.shape[0],
            self.action_dim,
        ), "Incorrect shape for action or log_prob."
        return action, log_prob


class Critic:
    def __init__(
        self,
        hidden_size: int,
        hidden_layers: int,
        critic_lr: float,
        state_dim: int = 3,
        action_dim: int = 1,
        num_critics: int = 2,
        device: torch.device = torch.device("cpu"),
    ):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.num_critics = num_critics

        self.model1 = None
        self.model2 = None
        self.optimizer = None

        self.setup_critic()

    def setup_critic(self):
        # changes here
        if self.model1 is None:
            self.model1 = NeuralNetwork(
                    input_dim=self.state_dim + self.action_dim,
                    output_dim=1,
                    hidden_size=self.hidden_size,
                    hidden_layers=self.hidden_layers,
                    activation="relu",
                ).to(self.device)

        if self.model2 is None:
            self.model2 = NeuralNetwork(
                input_dim=self.state_dim + self.action_dim,
                output_dim=1,
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_layers,
                activation="relu",
            ).to(self.device)

        critic_params = list(self.model1.parameters()) + list(self.model2.parameters())
        self.optimizer = optim.Adam(critic_params, lr=self.critic_lr)

class TrainableParameter:
    """
    This class could be used to define a trainable parameter in your method. You could find it
    useful if you try to implement the entropy temperature parameter for SAC algorithm.
    """

    def __init__(
        self,
        init_param: float,
        lr_param: float,
        train_param: bool,
        device: torch.device = torch.device("cpu"),
    ):
        self.log_param = torch.tensor(
            np.log(init_param), requires_grad=train_param, device=device
        )
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training,
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(
            self.min_buffer_size, self.max_buffer_size, self.device
        )
        self.actor = None
        self._actor_params = {
            "hidden_size": 256,
            "hidden_layers": 2,
            "actor_lr": 0.0003,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "device": self.device,
        }
        self._critic_params = {
            "hidden_size": 256,
            "hidden_layers": 2,
            "critic_lr": 0.0003,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "device": self.device,
        }
        self.critic = None

        self._value_net_params = {
            "input_dim": self.state_dim,
            "output_dim": 1,
            "hidden_size": self._critic_params["hidden_size"],
            "hidden_layers": self._critic_params["hidden_layers"],
            "activation": "relu",
        }
        self.value_base = None
        self.vb_optimizer = None
        self.value_target = None

        self.gamma = 0.99
        self.setup_agent()

    def setup_agent(self):
        # changes here
        if self.actor is None:
            self.actor = Actor(**self._actor_params)
        if self.critic is None:
            self.critic = Critic(**self._critic_params)
        if self.value_base is None:
            self.value_base = NeuralNetwork(**self._value_net_params).to(self.device)
            self.vb_optimizer = optim.Adam(self.value_base.parameters(), lr=0.0003)
        if self.value_target is None:
            self.value_target = NeuralNetwork(**self._value_net_params).to(self.device)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode.
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        action = self.actor.get_action_and_log_prob(torch.tensor(s), not train)[0]

        assert action.shape == (1,), "Incorrect action shape."
        assert isinstance(action, np.ndarray), "Action dtype must be np.ndarray"
        return action

    def run_value_update_step(self, loss: torch.Tensor):
        self.vb_optimizer.zero_grad()
        loss.mean().backward()
        self.vb_optimizer.step()

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        """
        This function takes in a object containing trainable parameters and an optimizer,
        and using a given loss, runs one step of gradient update. If you set up trainable parameters
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        """
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(
        self,
        base_net: NeuralNetwork,
        target_net: NeuralNetwork,
        tau: float,
        soft_update: bool,
    ):
        """
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        """
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(
                    param_target.data * (1.0 - tau) + param.data * tau
                )
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        """
        This function represents one training iteration for the agent. It samples a batch
        from the replay buffer,and then updates the policy and critic networks
        using the sampled batch.
        """
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)
        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        # Update Value Network
        # TODO: Investigate whether to use deterministic (MAP Estimate) or sample from the policy distribution
        action, log_prob = self.actor.get_action_and_log_prob(s_batch, False)
        sa_batch = torch.cat((s_batch, action), dim=1)
        q1 = self.critic.model1(sa_batch)
        q2 = self.critic.model2(sa_batch)
        min_critic = torch.minimum(q1, q2)
        target_loss = (1/2) * (self.value_base(s_batch) - (min_critic - log_prob)).pow(2).mean()
        self.run_value_update_step(target_loss)

        # change: Implement Critic(s) update here.
        sa_batch = torch.cat((s_batch, a_batch), dim=1)
        q1 = self.critic.model1(sa_batch)
        q2 = self.critic.model2(sa_batch)
        target = r_batch + self.gamma * self.value_target(s_prime_batch)
        critic_loss_1 = (1/2 * (q1 - target)).pow(2).mean()
        critic_loss_2 = (1/2 * (q2 - target)).pow(2).mean()
        self.run_gradient_update_step(self.critic, critic_loss_1)
        self.run_gradient_update_step(self.critic, critic_loss_2)

        # change: Implement Policy update here
        # Update Value Network
        action, log_prob = self.actor.get_action_and_log_prob(s_batch, False)
        # TODO: Add gaussian to the state input or the action output

        sa_batch = torch.cat((s_batch, action), dim=1)
        q1 = self.critic.model1(sa_batch)
        q2 = self.critic.model2(sa_batch)
        min_critic = torch.minimum(q1, q2)
        policy_loss = (log_prob - min_critic).mean()
        self.run_gradient_update_step(self.actor.model, policy_loss)

        self.critic_target_update(base_net=self.value_base, target_net=self.value_target, tau=0.005, soft_update=True)


# This main function is provided here to enable some basic testing.
# ANY changes here WON'T take any effect while grading.
if __name__ == "__main__":
    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print("\n")

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")

    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()

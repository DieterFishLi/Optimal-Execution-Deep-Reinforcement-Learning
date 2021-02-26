import random
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA =  torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.num_act = num_actions
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon, ):
        # if state[-1] == 2:
        #     action = env.inventory
        #     return action
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), )
            q_value = self.forward(state)
            action = int(q_value.max(1)[1].data[0])
        else:
            action = random.randrange(0, self.num_act)
        return action


class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DuelingDQN, self).__init__()
        self.num_act = num_actions
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def act(self, state, epsilon, ):

        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(state).unsqueeze(0), )
                q_value = self.forward(state)
                action = int(q_value.max(1)[1].data[0])
        else:
            action = random.randrange(0, self.num_act)
        return action



class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, std=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.std_init = std

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_f, in_f))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_f, in_f))
        self.register_buffer('weight_epsilon', tensor=torch.FloatTensor(out_f, in_f))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_f))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_f))
        self.register_buffer('bias_epsilon', tensor=torch.FloatTensor(out_f))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x





class NoisyDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(NoisyDQN, self).__init__()
        self.linear = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 32)
        self.noisy1 = NoisyLinear(32, 32)
        self.noisy2 = NoisyLinear(32, num_actions)
        self.num_act = num_actions

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        x = self.noisy1(x)
        x = self.noisy2(x)
        return x

    def act(self, state, eplison):
        if random.random() < eplison:
            action = random.randrange(0, self.num_act)
        else:
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action = int(q_value.max(1)[1].data[0])
        return action

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()




class NoisyDuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(NoisyDuelingDQN, self).__init__()
        self.num_act = num_actions
        self.state_value = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.state_value_noisy1 = NoisyLinear(32, 32)
        self.state_value_noisy2 = NoisyLinear(32, 1)

        self.advantage = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.advantage_noisy1 = NoisyLinear(32, 32)
        self.advantage_noisy2 = NoisyLinear(32, num_actions)

    def forward(self, x):
        state_value = self.state_value(x)
        state_value = self.state_value_noisy1(state_value)
        state_value = self.state_value_noisy2(state_value)

        advantage = self.advantage(x)
        advantage = self.advantage_noisy1(advantage)
        advantage = self.advantage_noisy2(advantage)
        return state_value + advantage - advantage.mean()

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0),)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(0, self.num_act)
        return action

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
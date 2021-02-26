from env import StockTrading, AMOUNT_TO_SELL
from buffer import ReplayBuffer

import torch
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from models import NoisyDuelingDQN, DuelingDQN, Variable, USE_CUDA
import random
import logging
logging.basicConfig(level=logging.INFO)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 20000

epsilon_by_step = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

env = StockTrading('000651.SZ')
in_put = len(env.reset())
# model = DQN(len(env.reset()), len(env.action_space))
model = NoisyDuelingDQN(in_put, len(env.action_space))

if USE_CUDA:
    model = model.cuda()

optimizer = optim.RMSprop(model.parameters())
replay_buffer = ReplayBuffer(50000)


def plot(step_idx, rewards, losses, slip):
    fig = plt.figure(figsize=(12, 3)) #figsize=(20, 5)
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (step_idx, np.mean(rewards[-20:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.subplot(133)
    plt.title('PnL over TWAP')
    plt.plot(slip)
    return fig

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + 0.99 * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

EXP = 'noisy_dueling_1003_pretrain_20000_sparsereward'

if __name__ == '__main__':
    num_steps = 100000
    batch_size = 128

    losses = []
    all_rewards = []
    all_slippage = [0]
    episode_reward = 0
    episode_slippage = 0
    state = env.reset()
    w = SummaryWriter('runs/' + EXP)
    w.add_graph(model, Variable(torch.FloatTensor(state).unsqueeze(0), ))
    for idx in range(4000):
        epsilon = 1
        action = random.randrange(env.twap, 2 * env.twap + 1)
        # logging.info('Sell vol %d' % action)
        next_state, reward, done = env.step(action)
        sup  = max(env.action_space)
        replay_buffer.push(state, action, reward, next_state, done)
        if done:
            w.add_figure('MKt', env.render(), idx)
            state = env.reset()
    logging.info('Pretrain Done')
    state = env.reset()

    for idx in range(1, num_steps + 1):
        epsilon = epsilon_by_step(idx)
        action = model.act(state, epsilon, )
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            w.add_figure('MKt', env.render(), idx)
            episode_slippage = env.slippage()
            all_slippage.append(episode_slippage)
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size and idx % (20) == 0:
            loss = compute_td_loss(batch_size)
            losses.append(loss.item())

        if idx % (200) == 0:
            logging.info('Current Step {0}'.format(idx))
            w.add_figure('Monitor', plot(idx, all_rewards, losses, all_slippage), idx)

    w.close()
    torch.save(model, EXP)
    print('Done!')
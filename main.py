from env import StockTrading, AMOUNT_TO_SELL
from buffer import ReplayBuffer

import torch
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from models import DQN, DuelingDQN, Variable, USE_CUDA
import random
from env import evualuate
import logging
import os
import math
logging.basicConfig(level=logging.INFO)

epsilon_start = 0.15
epsilon_final = 0.01
epsilon_decay = 20000

epsilon_by_step = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

env = StockTrading('000651.SZ')
env.reset()
model = DQN(len(env.reset()), len(env.action_space))
# model = DuelingDQN(len(env.reset()), len(env.action_space))

if USE_CUDA:
    model = model.cuda()

optimizer = optim.Adam(model.parameters())
replay_buffer = ReplayBuffer(20000)


def plot(step_idx, rewards, losses, slip):
    fig = plt.figure(figsize=(12, 3)) #figsize=(20, 5)
    plt.subplot(131)
    plt.title('Step: %s. reward: %s' % (step_idx, np.mean(rewards[-20:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.subplot(133)
    plt.title('PnL over TWAP')
    plt.plot(slip)
    return fig

def plot_test_loss(step_idx, losses):
    fig = plt.figure()
    plt.title('Step: %s, eva loss: %s', step_idx, np.mean(losses[-20:]))
    plt.plot(losses)
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

EXP = 'dqn_1008_pretrain_4000_sparsereward_80w'

if __name__ == '__main__':
    num_steps = 800000
    batch_size = 128

    losses = []
    all_rewards = []
    all_slippage = [0]
    all_vali_loss = []
    episode_reward = 0
    episode_slippage = 0
    state = env.reset()
    w = SummaryWriter('runs/' + EXP)
    w.add_graph(model, Variable(torch.FloatTensor(state).unsqueeze(0), ))


    for idx in range(1200):
        epsilon = 1
        action = random.randrange(env.twap, 2 * env.twap + 1)
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        if done:
            w.add_figure('MKt', env.render(), idx)
            state = env.reset()

    state = env.reset()
    for idx in range(200):
        action = env.twap
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        if done:
            w.add_figure('MKt', env.render(), idx)
            state = env.reset()

    state = env.reset()
    for idx in range(1666):
        action = 2 * env.twap
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        if done:
            w.add_figure('MKt', env.render(), idx)
            state = env.reset()

    state = env.reset()
    for idx in range(200):
        action = random.randrange(0, 2 * env.twap + 1)
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        if done:
            w.add_figure('MKt', env.render(), idx)
            state = env.reset()



    compute_td_loss(2048)
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
            all_rewards.append(episode_reward)
            episode_reward = 0
            state = env.reset()

        if len(replay_buffer) > batch_size and idx % (20) == 0:
            loss = compute_td_loss(batch_size)
            losses.append(loss.item())
            # if math.isclose(epsilon, epsilon_final):
            #     vali_loss, vali_slippage = evualuate(env, model, epsilon)
            #     all_vali_loss.append(vali_loss)
            #     logging.info('Vali Loss: %s, Vali avg Slippage: %s' % (vali_loss, np.mean(vali_slippage)))
            #     if abs(vali_loss) <= 1e-6:
            #         break

        if idx % (200) == 0:
            logging.info('Current Step {0}'.format(idx))
            w.add_figure('Monitor', plot(idx, all_rewards, losses, all_slippage), idx)

    w.close()
    torch.save(model, os.path.join('trained_models', EXP + '.h5'))
    print('Done!')
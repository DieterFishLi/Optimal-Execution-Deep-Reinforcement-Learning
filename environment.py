import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from render import MyFormatter
from collections import deque
from glob import glob
from dataloader import dataloader_V2
import random, math
import os
import pickle
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

AMOUNT_TO_SELL_L1 = {'600048.SH': 6000,  # 16311.020576131687
                     '600606.SH': 20000,  # 57234.91358024691
                     '601155.SH': 1400,  # 3715.9753086419755
                     '600519.SH': 200,  # 415.24691358024694
                     '603288.SH': 200,  # 506.6872427983539
                     '600690.SH': 3000,  # 8135.275720164609
                     '601398.SH': 290000,  # 850369.8600823046
                     '601939.SH': 60000,  # 173612.18106995884
                     '000001.SZ': 9878.823045267489,  # 26688.90534979424
                     '000333.SZ': 1200,  # 2557.971193415638
                     '000651.SZ': 60,  # 2939.1666666666665
                     '000661.SZ': 140,  # 274.24380165289256
                     '000538.SZ': 320,  # 657.3168724279835
                     '000002.SZ': 4300,  # 10075.271604938273
                     '001979.SZ': 1300,  # 3352.4504132231405
                     '000951.SZ': 900}  # 2163.3991769547324


def _slippage(price, ref_price):
    s = 10000 * (price - ref_price) / ref_price
    return s


class StockTrading(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, code):
        super(StockTrading, self).__init__()
        self.code = code
        try:
            train_file = open(os.path.join(code, 'train_file.dat'), 'rb')
            self.train_file_list = pickle.load(train_file)
        except:
            train_file = open(os.path.join(code, 'train_file.dat'), 'wb')
            df = deque([dataloader_V2(i) for i in glob(os.path.join(code, 'train', '*.h5'))])
            pickle.dump(df, train_file)
            train_file.close()
            train_file = open(os.path.join(code, 'train_file.dat'), 'rb')
            self.train_file_list = pickle.load(train_file)

        self.test_file_list = deque(glob(os.path.join(code, 'test', '*.h5')))
        logging.info('Train Files Loaded')


    def reset(self, mode='train'):
        if mode == 'train':
            self.cur = self.train_file_list.popleft()  # else
            self.train_file_list.append(self.cur)
        elif mode == 'vali':
            self.cur = random.sample(self.test_file_list, 1)[0]
        elif mode == 'test':
            self.cur = self.test_file_list.popleft()
        else:
            self.cur = self.train_file_list.popleft()

        self.df = self.cur
        self.twap = math.ceil(AMOUNT_TO_SELL_L1[self.code] / len(self.df))
        self.action_space = np.arange(0., 2 * (self.twap) + 1)
        self._inventory = self.inventory = AMOUNT_TO_SELL_L1[self.code]
        self._tick_left = self.tick_left = len(self.df)
        self.current_step = 0
        self.start_tick = self.df.iloc[0]
        self.trd_limit = pd.DataFrame(columns=['trd_time', 'trd_price', 'trd_vol', 'IS'])
        self.base_price = np.min([self.start_tick.BuyPrice01, self.start_tick.SellPrice01,])
        self.twap_limit = self._twap_limit()

        return self._next_observation()

    def _twap_limit(self):
        df = pd.DataFrame(columns=['trd_time', 'trd_price', 'trd_vol', 'IS'])
        inventory = self.inventory
        i = 0
        while inventory > 0:
            action = min(self.twap, self.inventory)
            bar = self.df.iloc[i]
            trd_time, trd_price, trd_vol, IS = self.mo_match(action, bar)
            inventory -= action
            df.loc[i] = [trd_time, trd_price, trd_vol, IS]
            i += 1
        return df

    def _IS(self, trd_price, trd_vol):
        return trd_vol * trd_price * 100 - trd_vol * self.base_price * 100

    def _next_observation(self):
        self.tick_left -= 1
        mkt_feature = self.df.iloc[self.current_step][['OrderImbalance', 'PriceDiff', ]].to_numpy().astype(
            'float32')
        buy_vol = self.df.iloc[self.current_step][['BuyVolume01',
                                                   'BuyVolume02',
                                                   'BuyVolume03',
                                                   'BuyVolume04',
                                                   'BuyVolume05', ]].astype('float32')
        buy_vol /= buy_vol.sum()

        sell_vol = self.df.iloc[self.current_step][['SellVolume05',
                                                    'SellVolume04',
                                                    'SellVolume03',
                                                    'SellVolume02',
                                                    'SellVolume01', ]].astype('float32')
        sell_vol /= sell_vol.sum()

        hist_buy_price = self.df.iloc[self.current_step][
            ['BuyPrice01_avg_20', 'BuyPrice01_avg_60', 'BuyPrice01_avg_100']].astype('float32')
        hist_buy_price /= hist_buy_price.sum()

        hist_sell_price = self.df.iloc[self.current_step][['SellPrice01_avg_20',
                                                           'SellPrice01_avg_60',
                                                           'SellPrice01_avg_100']].astype('float32')
        hist_sell_price /= hist_sell_price.sum()

        buy_price = self.df.iloc[self.current_step][['BuyPrice01',
                                                     'BuyPrice02',
                                                     'BuyPrice03',
                                                     'BuyPrice04',
                                                     'BuyPrice05']].astype('float32')
        buy_price /= buy_price.sum()

        sell_price = self.df.iloc[self.current_step][['SellPrice01',
                                                      'SellPrice02',
                                                      'SellPrice03',
                                                      'SellPrice04',
                                                      'SellPrice05']].astype('float32')
        sell_price /= self.df.iloc[0][['SellPrice01',
                                       'SellPrice02',
                                       'SellPrice03',
                                       'SellPrice04',
                                       'SellPrice05']].astype('float32')

        pvt_feature = np.array([ self.inventory / self._inventory, self.tick_left / self._tick_left]).astype(
            'float32')

        return np.concatenate(
            (mkt_feature, buy_price, buy_vol, sell_price, sell_vol, hist_buy_price, hist_sell_price, pvt_feature,), )

    def step(self, action):
        self.current_step += 1
        action = min(action, self.inventory)
        self.inventory -= action
        done = True if self.inventory == 0 or self.tick_left == 1 else False

        _bar = self.df.iloc[self.current_step - 1]
        trd_limit = self.mo_match(action, _bar)
        self.trd_limit.loc[len(self.trd_limit)] = trd_limit

        reward = self.reward(action)

        if self.tick_left == 1 and self.inventory > 0:
            _bar = self.df.iloc[-1]
            inventory_left = self.inventory
            trd_limit = self.mo_match(inventory_left, _bar)
            self.trd_limit.loc[len(self.trd_limit)] = trd_limit
            reward += self.reward(inventory_left)  # - 0.5 * inventory_left

        if done:
            slippage = self.slippage_done()
            if slippage > 0:
                reward += 1
            elif slippage < 0:
                reward -= 1
            else:
                reward += 0
        next_state = self._next_observation()
        return next_state, reward, done

    def mo_match(self, action, bar):
        if action == 0:
            return [bar.TradingTime, 0.0, 0.0, 0.0]
        trd_limit = []
        left = action
        for level in range(5):
            level += 1
            p, v = bar['BuyPrice0' + str(level)], bar['BuyVolume0' + str(level)]
            filled_qty = v if left >= v else left
            left -= filled_qty
            trd_limit.append([p, filled_qty])
            if left <= 0:
                break
        trd_limit = np.array(trd_limit)
        trd_price = trd_limit[:, 1] @ trd_limit[:, 0] / trd_limit[:, 1].sum()
        trd_vol = action
        trd_time = bar.TradingTime
        IS = self._IS(trd_price, trd_vol)
        return [trd_time, trd_price, trd_vol, IS]

    def reward(self, action):
        if action == 0:
            if self.twap_limit.IS.iloc[:self.current_step].sum() < 0:
                reward = -abs(0.01 * self.twap_limit.IS.iloc[self.current_step - 1]) - 0.1 * self.twap
            elif self.twap_limit.IS.iloc[self.current_step - 1] > 0:
                if self.inventory == self._inventory:
                    reward = - abs(0.01 * self.twap_limit.IS.iloc[self.current_step - 1]) - 0.1 * self.twap
                else:
                    reward = 0
            else:
                reward = abs(0.01 * self.twap_limit.IS.iloc[self.current_step - 1])
        else:
            reward = 0.1 * self.slippage_baseline()
            if self.twap_limit.IS.iloc[:self.current_step].sum() < 0:
                if action > self.twap:
                    reward += abs(0.01 * self.twap_limit.IS.iloc[self.current_step - 1])
                else:
                    reward -= abs(0.01 * self.twap_limit.IS.iloc[self.current_step - 1])
            elif self.twap_limit.IS.iloc[self.current_step - 1] > 0:
                if self.trd_limit.IS.iloc[-1] <= self.twap_limit.IS.iloc[self.current_step - 1]:
                    reward -= 0.1 * abs(action - self.twap)
                else:
                    reward += 0.1 * abs(action - self.twap)
            else:
                reward -= 0
        return reward

    def slippage(self):
        price = (self.trd_limit.trd_price @ self.trd_limit.trd_vol) / self.trd_limit.trd_vol.sum()
        ref_price = self.df.set_index('TradingTime').reindex(self.trd_limit.trd_time).BuyPrice01.mean()
        return _slippage(price, ref_price)

    def slippage_baseline(self):
        price = (self.trd_limit.trd_price @ self.trd_limit.trd_vol) / self.trd_limit.trd_vol.sum()
        ref = self.twap_limit.set_index('trd_time').reindex(self.trd_limit.trd_time)
        ref_price = ref.trd_price
        ref_vol = ref.trd_vol
        ref_price = (ref_price @ ref_vol) / ref_vol.sum()
        return _slippage(price, ref_price)

    def slippage_done(self):
        price = (self.trd_limit.trd_price @ self.trd_limit.trd_vol) / self.trd_limit.trd_vol.sum()
        ref = self.twap_limit
        ref_price = ref.trd_price
        ref_vol = ref.trd_vol
        ref_price = (ref_price @ ref_vol) / ref_vol.sum()
        return _slippage(price, ref_price)

    def render(self, mode='human'):
        return [self.twap_limit, self.trd_limit]



if __name__ == '__main__':
    env = StockTrading('000651.SZ')

# Order-Placement-With-Deep-Reinforcement-Learning

This project focuses on trading execution within a short time horizon. 
The trading instruction is to sell Q number of shares of XYZ stock within 1 minute from 9:31 to 9:32, 
which is the beginning of the continuous trading session of a trading day in China. The chosen benchmark is the TWAP algorithm. 
The Q is set to the level ensuring that the trading algorithm has its role to play. 
If the Q were too small, the trading task would become too easy that it is impossible to achieve a better result. 
If the Q were too large, the trade task would consume a lot of liquidity on the market making the benchmark price an unreachable upper bound.

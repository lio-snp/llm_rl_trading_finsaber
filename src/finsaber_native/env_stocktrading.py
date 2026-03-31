from __future__ import annotations

from typing import Callable, List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradingEnv(gym.Env):
    """
    A stock trading environment for OpenAI gym

    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        hmax (int): Maximum cash to be traded in each trade per asset.
        initial_amount (int): Amount of cash initially available
        buy_cost_pct (float, array): Cost for buying shares, each index corresponds to each asset
        sell_cost_pct (float, array): Cost for selling shares, each index corresponds to each asset
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        revise_state: Callable | None = None,
        intrinsic_reward: Callable | None = None,
        use_revised_for_policy: bool = False,
        use_intrinsic: bool = False,
        intrinsic_w: float = 0.0,
        intrinsic_scale_mode: str = "raw",
        intrinsic_timing: str = "pre_action_state",
        intrinsic_input_mode: str = "raw",
        policy_state_fn: Callable | None = None,
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.revise_state = revise_state
        self.intrinsic_reward = intrinsic_reward
        self.use_revised_for_policy = bool(use_revised_for_policy)
        self.use_intrinsic = bool(use_intrinsic)
        self.intrinsic_w = float(intrinsic_w)
        self.intrinsic_scale_mode = str(intrinsic_scale_mode or "raw")
        self.intrinsic_timing = str(intrinsic_timing or "pre_action_state")
        self.intrinsic_input_mode = str(intrinsic_input_mode or "raw").lower()
        self.policy_state_fn = policy_state_fn
        # initalize state
        self.state = self._initiate_state()
        initial_obs = self._policy_observation(self.state)
        self._last_observation = np.asarray(initial_obs, dtype=np.float32).reshape(-1)

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.actions_policy_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        self.intrinsic_values = []
        self.reward_env_values = []
        self.reward_total_values = []
        self.action_penalty_values = []
        self.portfolio_weight_values = []
        self.cash_weight_values = []
        self.portfolio_weight_change_values = []
        initial_weights, initial_cash_weight = self._portfolio_weights_from_state(self.state)
        self._last_portfolio_weights = np.asarray(initial_weights, dtype=np.float32).reshape(-1)
        self._last_cash_weight = float(initial_cash_weight)
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(self._last_observation.shape[0]),)
        )

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)
            info = self._build_step_info(
                reward_env=float(self.reward),
                intrinsic=0.0,
                reward_total=float(self.reward),
                action_policy=np.zeros(self.stock_dim, dtype=float),
                action_executed=np.zeros(self.stock_dim, dtype=float),
                portfolio_value=float(end_total_asset),
                portfolio_weights=self._last_portfolio_weights,
                cash_weight=float(self._last_cash_weight),
                portfolio_weight_change=0.0,
            )
            return self._policy_observation(self.state), self.reward, self.terminal, False, info

        else:
            prev_state = np.asarray(self.state, dtype=np.float32).reshape(-1)
            action_policy = np.asarray(actions, dtype=float).reshape(-1)
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_policy_memory.append(np.asarray(action_policy, dtype=float).reshape(-1))
            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            reward_env_raw = end_total_asset - begin_total_asset
            self.rewards_memory.append(reward_env_raw)
            reward_env_scaled = reward_env_raw * self.reward_scaling
            intrinsic_val = 0.0
            if self.use_intrinsic and self.intrinsic_reward is not None:
                intrinsic_source_state = prev_state
                if str(self.intrinsic_timing).lower() == "post_action_state":
                    intrinsic_source_state = np.asarray(self.state, dtype=np.float32).reshape(-1)
                intrinsic_input = self._intrinsic_input_state(intrinsic_source_state)
                try:
                    intrinsic_val = float(self.intrinsic_reward(intrinsic_input))
                except Exception:
                    intrinsic_val = 0.0
                intrinsic_val = self._scale_intrinsic(intrinsic_val)
            self.reward = float(reward_env_scaled + self.intrinsic_w * intrinsic_val)
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step
            portfolio_weights, cash_weight = self._portfolio_weights_from_state(self.state)
            weight_change = float(
                np.sum(np.abs(np.asarray(portfolio_weights, dtype=float) - self._last_portfolio_weights))
            ) if self._last_portfolio_weights.size == len(portfolio_weights) else 0.0
            self._last_portfolio_weights = np.asarray(portfolio_weights, dtype=np.float32).reshape(-1)
            self._last_cash_weight = float(cash_weight)
            self.intrinsic_values.append(float(intrinsic_val))
            self.reward_env_values.append(float(reward_env_scaled))
            self.reward_total_values.append(float(self.reward))
            self.action_penalty_values.append(0.0)
            self.portfolio_weight_values.append(list(portfolio_weights))
            self.cash_weight_values.append(float(cash_weight))
            self.portfolio_weight_change_values.append(float(weight_change))
            info = self._build_step_info(
                reward_env=float(reward_env_scaled),
                intrinsic=float(intrinsic_val),
                reward_total=float(self.reward),
                action_policy=action_policy,
                action_executed=np.asarray(actions, dtype=float).reshape(-1),
                portfolio_value=float(end_total_asset),
                portfolio_weights=portfolio_weights,
                cash_weight=float(cash_weight),
                portfolio_weight_change=float(weight_change),
            )

        return self._policy_observation(self.state), self.reward, self.terminal, False, info

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.actions_policy_memory = []
        self.date_memory = [self._get_date()]
        self.intrinsic_values = []
        self.reward_env_values = []
        self.reward_total_values = []
        self.action_penalty_values = []
        self.portfolio_weight_values = []
        self.cash_weight_values = []
        self.portfolio_weight_change_values = []
        initial_weights, initial_cash_weight = self._portfolio_weights_from_state(self.state)
        self._last_portfolio_weights = np.asarray(initial_weights, dtype=np.float32).reshape(-1)
        self._last_cash_weight = float(initial_cash_weight)
        self._last_observation = self._policy_observation(self.state)

        self.episode += 1

        return self._policy_observation(self.state), {}

    def render(self, mode="human", close=False):
        return self.state

    def _sanitize_state(self, state) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_revise_state(self, state) -> np.ndarray:
        raw = self._sanitize_state(state)
        if self.revise_state is None:
            return raw
        try:
            revised = self.revise_state(raw)
            if revised is None:
                return raw
            return self._sanitize_state(revised)
        except Exception:
            return raw

    def _policy_observation(self, state) -> np.ndarray:
        raw = self._sanitize_state(state)
        if self.policy_state_fn is not None:
            try:
                return self._sanitize_state(self.policy_state_fn(raw))
            except Exception:
                return raw
        if self.use_revised_for_policy:
            return self._apply_revise_state(raw)
        return raw

    def _intrinsic_input_state(self, state) -> np.ndarray:
        raw = self._sanitize_state(state)
        if self.intrinsic_input_mode == "revised":
            return self._apply_revise_state(raw)
        return raw

    def _scale_intrinsic(self, value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        mode = self.intrinsic_scale_mode.lower()
        if mode == "bounded_100":
            return float(np.clip(value, -100.0, 100.0))
        if mode == "normalized":
            return float(np.tanh(value) * 100.0)
        return float(value)

    def _portfolio_weights_from_state(self, state) -> tuple[list[float], float]:
        raw = self._sanitize_state(state)
        prices = raw[1 : 1 + self.stock_dim]
        holdings = raw[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
        asset_values = prices * holdings
        cash = float(raw[0]) if raw.size > 0 else 0.0
        total_value = float(cash + np.sum(asset_values))
        if total_value <= 0:
            return [0.0] * self.stock_dim, 0.0
        return (
            np.asarray(asset_values / total_value, dtype=float).reshape(-1).tolist(),
            float(cash / total_value),
        )

    def _build_step_info(
        self,
        *,
        reward_env: float,
        intrinsic: float,
        reward_total: float,
        action_policy,
        action_executed,
        portfolio_value: float,
        portfolio_weights,
        cash_weight: float,
        portfolio_weight_change: float,
    ) -> dict:
        return {
            "reward_env": float(reward_env),
            "intrinsic": float(intrinsic),
            "reward_total": float(reward_total),
            "action_policy": np.asarray(action_policy, dtype=float).reshape(-1).tolist(),
            "action_executed": np.asarray(action_executed, dtype=float).reshape(-1).tolist(),
            "portfolio_value": float(portfolio_value),
            "portfolio_weights": np.asarray(portfolio_weights, dtype=float).reshape(-1).tolist(),
            "cash_weight": float(cash_weight),
            "portfolio_weight_change": float(portfolio_weight_change),
            "state_dim": int(self._policy_observation(self.state).shape[0]),
        }

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def save_lesr_trace_bundle(self):
        return {
            "rewards_memory": [float(x) for x in list(self.rewards_memory)],
            "asset_memory": [float(x) for x in list(self.asset_memory)],
            "state_memory": [list(np.asarray(x, dtype=float).reshape(-1)) for x in list(self.state_memory)],
            "actions_policy_memory": [
                [float(v) for v in list(np.asarray(x, dtype=float).reshape(-1))]
                for x in list(self.actions_policy_memory)
            ],
            "reward_env_values": [float(x) for x in list(self.reward_env_values)],
            "intrinsic_values": [float(x) for x in list(self.intrinsic_values)],
            "reward_total_values": [float(x) for x in list(self.reward_total_values)],
            "action_penalty_values": [float(x) for x in list(self.action_penalty_values)],
            "portfolio_weight_values": [
                [float(v) for v in list(np.asarray(x, dtype=float).reshape(-1))]
                for x in list(self.portfolio_weight_values)
            ],
            "cash_weight_values": [float(x) for x in list(self.cash_weight_values)],
            "portfolio_weight_change_values": [float(x) for x in list(self.portfolio_weight_change_values)],
        }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

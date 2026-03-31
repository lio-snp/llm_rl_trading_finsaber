import datetime
import pandas as pd
import numpy as np
import itertools
from types import SimpleNamespace
try:
    from backtest.strategy.timing.base_strategy import BaseStrategy
    from backtest.finsaber_bt import FINSABERBt
    from backtest.toolkit.operation_utils import aggregate_results_one_strategy
except Exception:
    BaseStrategy = object
    FINSABERBt = None
    aggregate_results_one_strategy = None

# FinRL imports
from src.finsaber_native.models import DRLAgent
from src.finsaber_native.env_stocktrading import StockTradingEnv
from src.finsaber_native.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from src.finsaber_native.preprocessors import FeatureEngineer

from stable_baselines3.common.noise import (
    OrnsteinUhlenbeckActionNoise,
    NormalActionNoise,
)

class FinRLStrategy(BaseStrategy):
    params = (
        ("algorithm", "td3"),           # Options: A2C, PPO, SAC, TD3
        ("total_timesteps", 5000),
        ("initial_amount", 100000),
        ("total_days", 0),
        ("train_period", 252 * 3),
        # ("train_period", 252),
    )

    def __init__(self, train_data: pd.DataFrame, strat_params=None):
        super().__init__()
        # print(f"RL Strategy initialised with {self.params.algorithm} algorithm.")
        if train_data is None:
            raise ValueError("Train data must be provided.")

        self.model_params = {
            "sac": {
                "learning_rate": 2e-2,
                "buffer_size": 1_000_000,
                "batch_size": 256,
                "learning_starts": 100,
                "ent_coef": 0.1,  # let the algorithm tune exploration
                "tau": 0.005,  # soft‐update rate
                "gamma": 0.99,  # discount factor
                "action_noise": "normal"
            },
            "ppo": {
                "n_steps": 2048,
                "batch_size": 64,  # smaller minibatches ⇒ more updates per rollout
                "n_epochs": 10,  # more policy passes
                "learning_rate": 2.5e-4,
                "ent_coef": 0.1,  # optional small entropy bonus
                "clip_range": 0.2,
                "gae_lambda": 0.95,
                "gamma": 0.99,
            },
            "a2c": {
                "n_steps": 100,
                "learning_rate": 1e-5,
                "ent_coef": 0.1,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "gae_lambda": 0.95,
                "gamma": 0.99,
            },
            "ddpg": {
                "learning_rate": 2e-2,
                "batch_size": 256,
                "buffer_size": 1_000_000,
                "tau": 0.005,
                "gamma": 0.99,

            },
            "td3": {
                "learning_rate": 3e-2,
                "buffer_size": 1_000_000,
                "tau": 0.005,
                "gamma": 0.99,
                "policy_delay": 2,
                "target_policy_noise": 0.5,
                "target_noise_clip": 0.5,
                "action_noise": "normal"
            },
        }

        self.raw_train_data = train_data.copy()
        # Convert multi-index raw data to long format suitable for FeatureEngineer
        self.formatted_raw = self.format_raw_data_for_fe(self.raw_train_data)
        # Preprocess using FinRL's FeatureEngineer
        self.train_data = self.preprocess_data(self.formatted_raw)

        self.test_data = []
        self.df_account_value, self.df_actions = [], []

        if self.train_data.empty:
            raise ValueError("Preprocessed train data is empty.")

        self.model = self.train_drl_model(
            algorithm=self.params.algorithm,
            total_timesteps=self.params.total_timesteps
        )

        self.history = {tic: [] for tic in self.train_data["tic"].unique()}

    def format_raw_data_for_fe(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw_data from wide MultiIndex format to a long DataFrame with single-level columns.
        Expected raw_data:
            - Index: dates
            - Columns: MultiIndex with levels: [feature, symbol]
        Resulting DataFrame will have columns: date, open, high, low, close, ... and a 'tic' column.
        """
        # Reset index to bring dates into a column.
        df = raw_data.copy().reset_index()
        ticker = df.columns[1][1]
        df.columns = df.columns.droplevel(1)

        # If the reset index column is not named 'date', rename it.
        if df.columns[0] != "date":
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

        df["tic"] = ticker
        df["day"] = df["date"].dt.dayofweek

        # drop nan rows
        df = df.dropna()

        return df

    def preprocess_data(self, formatted_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess formatted raw data using FinRL's FeatureEngineer.
        Expects formatted_raw to have columns: date, tic, open, high, low, close, etc.
        """
        try:
            fe = FeatureEngineer(
                use_technical_indicator=True,
                tech_indicator_list=INDICATORS,
                use_vix=False,
                use_turbulence=True,
                user_defined_feature=False
            )
            return fe.preprocess_data(formatted_raw)

        except Exception:
            print("Failed to add turbulence index. Proceeding without it.")
            fe = FeatureEngineer(
                use_technical_indicator=True,
                tech_indicator_list=INDICATORS,
                use_vix=False,
                use_turbulence=False,
                user_defined_feature=False
            )
            return fe.preprocess_data(formatted_raw)

    def build_default_env_kwargs(self, df_env: pd.DataFrame, env_kwargs: dict = None):
        stock_dim = len(df_env["tic"].unique())
        state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim

        default_env_kwargs = {
            "hmax": 1000,
            "initial_amount": self.params.initial_amount,
            "num_stock_shares": [0] * stock_dim,
            "buy_cost_pct": [0.0049] * stock_dim,
            "sell_cost_pct": [0.0049] * stock_dim,
            "state_space": state_space,
            "stock_dim": stock_dim,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dim,
            "reward_scaling": 1e-4,
        }

        if env_kwargs:
            default_env_kwargs.update(env_kwargs)
        return default_env_kwargs

    def prepare_test_data_frames(self, test_frames, history_frame: pd.DataFrame | None = None):
        self.test_data = []
        history_frame = self.formatted_raw if history_frame is None else history_frame.copy()
        history_frame["date"] = pd.to_datetime(history_frame["date"])
        history_frame = history_frame.sort_values(["date", "tic"]).reset_index(drop=True)

        for frame in test_frames:
            test_df = frame.copy()
            test_df["date"] = pd.to_datetime(test_df["date"])
            test_start_date = test_df["date"].min()
            test_end_date = test_df["date"].max()
            merged = pd.concat([history_frame, test_df], ignore_index=True)
            merged = self.preprocess_data(merged)
            merged = merged[(merged["date"] >= test_start_date) & (merged["date"] <= test_end_date)]
            merged = merged.sort_values(["date", "tic"]).reset_index(drop=True)
            merged.index = merged["date"].factorize()[0]
            self.test_data.append(merged)

    def predict_test_data(self, trained_model, default_env_kwargs: dict, deterministic: bool = True):
        self.df_account_value, self.df_actions = [], []
        self.trade_envs = []
        for test_df in self.test_data:
            e_trade = StockTradingEnv(df=test_df, **default_env_kwargs)
            df_account_value, df_actions = DRLAgent.DRL_prediction(
                model=trained_model,
                environment=e_trade,
                deterministic=deterministic
            )

            df_account_value["date"] = pd.to_datetime(df_account_value["date"]).dt.date
            if "date" not in df_actions.columns:
                df_actions = df_actions.reset_index()
            df_actions["date"] = pd.to_datetime(df_actions["date"]).dt.date

            self.df_account_value.append(df_account_value)
            self.df_actions.append(df_actions)
            self.trade_envs.append(e_trade)

    def train_drl_model(
        self,
        algorithm: str = "a2c",
        total_timesteps: int = 50000,
        env_kwargs: dict = None,
        *,
        seed: int | None = None,
        model_kwargs_override: dict | None = None,
        test_frames=None,
        history_frame: pd.DataFrame | None = None,
        deterministic: bool = True,
    ):
        """
        Train a DRL agent using FinRL.
        """
        df_env = self.train_data.copy()
        default_env_kwargs = self.build_default_env_kwargs(df_env, env_kwargs=env_kwargs)
        e_train = StockTradingEnv(df=df_env, **default_env_kwargs)
        env_train, _ = e_train.get_sb_env()
        agent = DRLAgent(env=env_train)

        model_kwargs = dict(self.model_params[algorithm])
        if model_kwargs_override:
            model_kwargs.update(model_kwargs_override)

        model = agent.get_model(
            algorithm,
            verbose=0,
            seed=seed,
            model_kwargs=model_kwargs,
        )
        trained_model = agent.train_model(
            model=model,
            tb_log_name=algorithm,
            total_timesteps=total_timesteps)

        if test_frames is None:
            for test_d in self.datas:
                test_df = test_d._dataname
                test_start_date = test_df.index[0]
                test_end_date = test_df.index[-1]

                test_df = test_df.rename_axis('date').reset_index()
                test_df['tic'] = test_d._name

                test_df = pd.concat([self.formatted_raw, test_df], ignore_index=True)
                test_df = self.preprocess_data(test_df)
                test_df = test_df[(test_df['date'] >= test_start_date) & (test_df['date'] <= test_end_date)]
                test_df = test_df.sort_values(['date', 'tic']).reset_index(drop=True)

                self.test_data.append(test_df)
        else:
            self.prepare_test_data_frames(test_frames, history_frame=history_frame)

        self.predict_test_data(trained_model, default_env_kwargs, deterministic=deterministic)

        return trained_model, default_env_kwargs

    @classmethod
    def create_standalone(
        cls,
        *,
        train_data: pd.DataFrame,
        algorithm: str,
        total_timesteps: int,
        initial_amount: float,
    ):
        obj = cls.__new__(cls)
        try:
            BaseStrategy.__init__(obj)
        except Exception:
            pass
        obj.params = SimpleNamespace(
            algorithm=str(algorithm).lower(),
            total_timesteps=int(total_timesteps),
            initial_amount=float(initial_amount),
            total_days=0,
            train_period=252 * 3,
        )
        obj.model_params = {
            "sac": {
                "learning_rate": 2e-2,
                "buffer_size": 1_000_000,
                "batch_size": 256,
                "learning_starts": 100,
                "ent_coef": 0.1,
                "tau": 0.005,
                "gamma": 0.99,
                "action_noise": "normal"
            },
            "ppo": {
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "learning_rate": 2.5e-4,
                "ent_coef": 0.1,
                "clip_range": 0.2,
                "gae_lambda": 0.95,
                "gamma": 0.99,
            },
            "a2c": {
                "n_steps": 100,
                "learning_rate": 1e-5,
                "ent_coef": 0.1,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "gae_lambda": 0.95,
                "gamma": 0.99,
            },
            "ddpg": {
                "learning_rate": 2e-2,
                "batch_size": 256,
                "buffer_size": 1_000_000,
                "tau": 0.005,
                "gamma": 0.99,
            },
            "td3": {
                "learning_rate": 3e-2,
                "buffer_size": 1_000_000,
                "tau": 0.005,
                "gamma": 0.99,
                "policy_delay": 2,
                "target_policy_noise": 0.5,
                "target_noise_clip": 0.5,
                "action_noise": "normal"
            },
        }
        obj.raw_train_data = train_data.copy()
        obj.formatted_raw = train_data.copy()
        obj.train_data = train_data.copy()
        obj.test_data = []
        obj.df_account_value, obj.df_actions = [], []
        obj.trade_envs = []
        obj.history = {tic: [] for tic in obj.train_data["tic"].unique()} if "tic" in obj.train_data.columns else {}
        obj.datas = []
        return obj

    def compute_state(self):
        state = []
        for d, test_df in zip(self.datas, self.test_data):
            today = d.datetime.date(0)
            tic = d._name

            # assert the ticker name
            assert tic == test_df["tic"].values[0]

            # get today's indicator features
            today_features = test_df[test_df['date'] == today]
            # drop columns that are not features
            today_features = today_features.drop(columns=["date", "tic", "day"])

            state.append(today_features)

        return np.array(state)


    def next(self):
        """
        Predict actions using the trained DRL model and execute trades.
        """

        for d, actions in zip(self.datas, self.df_actions):
            today = d.datetime.date(0)

            if today > actions['date'].tolist()[-1]:
                return

            try:
                act = int(actions[actions["date"] == today]["actions"].values[0])
            except Exception as e:
                print(f"No action detected on {today}: {e}")
                return

            # import pdb; pdb.set_trace()

            price = d.close[0]

            if act > 0:
                # check if we have enough cash to buy
                size = self._adjust_size_for_commission(min(int(self.broker.cash / price), act))
                self.buy(data=d, size=size)
                self.buys.append(today)
                self.trades.append(
                    {
                        "date": today,
                        "ticker": d._name,
                        "action": "buy",
                        "size": size,
                        "price": price,
                    }
                )
            elif act < 0:
                pos = self.getposition(data=d)
                size = self._adjust_size_for_commission(min(pos.size, -act))
                self.sell(data=d, size=size)
                self.sells.append(today)
                self.trades.append(
                    {
                        "date": today,
                        "action": "sell",
                        "size": size,
                        "price": price,
                    }
                )
            else:
                pass

        self.post_next_actions()


if __name__ == "__main__":
    # trade_config = {
    #     "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
    #     "silence": False,
    #     "setup_name": "selected_5",
    # }
    trade_config = {
        "date_from": "2022-10-06",
        "date_to": "2023-04-10",
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        "setup_name": "cherry_pick_both_finmem",
    }
    operator = FINSABERBt(trade_config)
    operator.run_iterative_tickers(FinRLStrategy)
    # operator.run_rolling_window(FinRLStrategy)

    aggregate_results_one_strategy(trade_config["setup_name"], FinRLStrategy.__name__)

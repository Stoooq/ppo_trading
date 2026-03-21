import matplotlib.pyplot as plt
import numpy as np
import torch


class SimpleTradingEnv:
    def __init__(self, window_size: int = 10, total_steps: int = 1000, seed: int = 42):
        self.window_size = window_size
        self.total_steps = total_steps
        self.seed = seed

        self.current_step = 0
        self.data = self._generate_data()
        self.current_position = 0.0

    def _generate_data(self) -> torch.Tensor:
        rng = np.random.default_rng(self.seed)
        t = np.arange(0, 20 * np.pi, 0.1)

        sine_wave = 0.4 * np.sin(t * 0.5)

        step_changes = rng.normal(loc=0.0, scale=0.05, size=len(t))
        random_walk = np.cumsum(step_changes)

        drift = t * 0.02

        data = sine_wave + random_walk + drift

        micro_noise = rng.normal(loc=0.0, scale=0.02, size=len(t))
        data = data + micro_noise

        data = data - np.min(data) + 10.0

        data = torch.from_numpy(data).float()

        plt.figure(figsize=(10, 5))
        plt.plot(data.numpy())
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Trading Data (Flattened Sine + Random Walk + Drift)")
        plt.grid(True)
        plt.show()

        return data

    def _get_state(self) -> torch.Tensor:
        start_idx = self.current_step
        end_idx = self.current_step + self.window_size

        state = self.data[start_idx:end_idx]

        return state

    def reset(self) -> torch.Tensor:
        self.current_step = 0
        self.current_position = 0.0

        return self._get_state()

    def step(self, action: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
        positions = [-1.0, 0.0, 1.0]
        self.current_position = positions[action]

        current_price = self.data[self.current_step + self.window_size - 1]

        self.current_step += 1

        done = (self.current_step + self.window_size) >= len(self.data)

        if not done:
            next_price = self.data[self.current_step + self.window_size - 1]
            price_diff = next_price - current_price
            reward_val = price_diff * self.current_position
            next_state = self._get_state()
        else:
            reward_val = 0.0
            self.current_step -= 1
            next_state = self._get_state()

        reward = torch.tensor(reward_val, dtype=torch.float32)

        return next_state, reward, done

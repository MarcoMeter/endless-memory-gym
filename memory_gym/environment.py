import gymnasium as gym

class CustomEnv(gym.Env):
    """"Extends the gym.Env class with a has_ground_truth_info attribute. This attribute is used to determine if the
    environment provides ground truth information using the info dictionary."""
    def __init__(self) -> None:
        self.has_ground_truth_info = False
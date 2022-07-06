from gym.envs.registration import register
from memory_gym.searing_spotlights.searing_spotlights import SearingSpotlightsEnv

register(
     id="SearingSpotlights-v0",
     entry_point="memory_gym.searing_spotlights.searing_spotlights:SearingSpotlightsEnv",
     max_episode_steps=1000,
)
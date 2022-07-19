from gym.envs.registration import register
from memory_gym.searing_spotlights import SearingSpotlightsEnv
from memory_gym.mortar_mayhem import MortarMayhemEnv

register(
     id="SearingSpotlights-v0",
     entry_point="memory_gym.searing_spotlights:SearingSpotlightsEnv",
     max_episode_steps=1000,
)

register(
     id="MortarMayhem-v0",
     entry_point="memory_gym.mortar_mayhem:MortarMayhemEnv",
     max_episode_steps=1000,
)
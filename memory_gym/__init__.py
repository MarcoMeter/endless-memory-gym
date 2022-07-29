from gym.envs.registration import register
from memory_gym.searing_spotlights import SearingSpotlightsEnv
from memory_gym.mortar_mayhem import MortarMayhemEnv
from memory_gym.mortar_mayhem_b import MortarMayhemTaskBEnv
from memory_gym.mystery_path import MysteryPathEnv

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

register(
     id="MortarMayhemB-v0",
     entry_point="memory_gym.mortar_mayhem_b:MortarMayhemTaskBEnv",
     max_episode_steps=1000,
)

register(
     id="MysteryPath-v0",
     entry_point="memory_gym.mystery_path:MysteryPathEnv",
     max_episode_steps=1000,
)
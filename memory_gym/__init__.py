from gym.envs.registration import register
from memory_gym.searing_spotlights import SearingSpotlightsEnv
from memory_gym.searing_spotlights_sanity import SearingSpotlightsSanityEnv
from memory_gym.mortar_mayhem import MortarMayhemEnv
from memory_gym.mortar_mayhem_grid import GridMortarMayhemEnv
from memory_gym.mortar_mayhem_b import MortarMayhemTaskBEnv
from memory_gym.mortar_mayhem_b_grid import GridMortarMayhemTaskBEnv
from memory_gym.mystery_path import MysteryPathEnv
from memory_gym.mystery_path_grid import GridMysteryPathEnv

register(
     id="SearingSpotlights-v0",
     entry_point="memory_gym.searing_spotlights:SearingSpotlightsEnv",
     max_episode_steps=512,
)

register(
     id="Sanity-v0",
     entry_point="memory_gym.searing_spotlights_sanity:SearingSpotlightsSanityEnv",
     max_episode_steps=512,
)

register(
     id="MortarMayhem-v0",
     entry_point="memory_gym.mortar_mayhem:MortarMayhemEnv",
     max_episode_steps=512,
)

register(
     id="MortarMayhem-Grid-v0",
     entry_point="memory_gym.mortar_mayhem_grid:GridMortarMayhemEnv",
     max_episode_steps=512,
)

register(
     id="MortarMayhemB-v0",
     entry_point="memory_gym.mortar_mayhem_b:MortarMayhemTaskBEnv",
     max_episode_steps=512,
)

register(
     id="MortarMayhemB-Grid-v0",
     entry_point="memory_gym.mortar_mayhem_b_grid:GridMortarMayhemTaskBEnv",
     max_episode_steps=512,
)

register(
     id="MysteryPath-v0",
     entry_point="memory_gym.mystery_path:MysteryPathEnv",
     max_episode_steps=512,
)

register(
     id="MysteryPath-Grid-v0",
     entry_point="memory_gym.mystery_path_grid:GridMysteryPathEnv",
     max_episode_steps=128,
)
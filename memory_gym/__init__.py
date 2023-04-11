from gymnasium.envs.registration import register
from memory_gym.searing_spotlights import SearingSpotlightsEnv
from memory_gym.searing_spotlights_gt import SearingSpotlightsGTEnv
from memory_gym.endless_searing_spotlights import EndlessSearingSpotlightsEnv
from memory_gym.mortar_mayhem import MortarMayhemEnv
from memory_gym.endless_mortar_mayhem import EndlessMortarMayhemEnv
from memory_gym.mortar_mayhem_grid import GridMortarMayhemEnv
from memory_gym.mortar_mayhem_b import MortarMayhemTaskBEnv
from memory_gym.mortar_mayhem_b_grid import GridMortarMayhemTaskBEnv
from memory_gym.mystery_path import MysteryPathEnv
from memory_gym.endless_mystery_path import EndlessMysteryPathEnv
from memory_gym.mystery_path_grid import GridMysteryPathEnv

register(
     id="SearingSpotlights-v0",
     entry_point="memory_gym.searing_spotlights:SearingSpotlightsEnv",
)

register(
     id="GroundTruth-SearingSpotlights-v0",
     entry_point="memory_gym.searing_spotlights_gt:SearingSpotlightsGTEnv",
)

register(
     id="Endless-SearingSpotlights-v0",
     entry_point="memory_gym.endless_searing_spotlights:EndlessSearingSpotlightsEnv",
)

register(
     id="Sanity-v0",
     entry_point="memory_gym.searing_spotlights_sanity:SearingSpotlightsSanityEnv",
)

register(
     id="MortarMayhem-v0",
     entry_point="memory_gym.mortar_mayhem:MortarMayhemEnv",
)

register(
     id="Endless-MortarMayhem-v0",
     entry_point="memory_gym.endless_mortar_mayhem:EndlessMortarMayhemEnv",
)

register(
     id="MortarMayhem-Grid-v0",
     entry_point="memory_gym.mortar_mayhem_grid:GridMortarMayhemEnv",
)

register(
     id="MortarMayhemB-v0",
     entry_point="memory_gym.mortar_mayhem_b:MortarMayhemTaskBEnv",
)

register(
     id="MortarMayhemB-Grid-v0",
     entry_point="memory_gym.mortar_mayhem_b_grid:GridMortarMayhemTaskBEnv",
)

register(
     id="MysteryPath-v0",
     entry_point="memory_gym.mystery_path:MysteryPathEnv",
)

register(
     id="Endless-MysteryPath-v0",
     entry_point="memory_gym.endless_mystery_path:EndlessMysteryPathEnv",
)

register(
     id="MysteryPath-Grid-v0",
     entry_point="memory_gym.mystery_path_grid:GridMysteryPathEnv",
)
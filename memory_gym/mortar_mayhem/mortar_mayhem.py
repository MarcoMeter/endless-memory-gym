import gym
import math
import numpy as np
import os
import pygame

from argparse import ArgumentParser
from gym import  spaces
from memory_gym.pygame_assets import CharacterController, Command, MortarTile
from pygame._sdl2 import Window, Texture, Renderer

SCALE = 1.0

class MortarMayhemEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "debug_rgb_array"],
        "render_fps": 25,
    }

    default_reset_parameters = {
                "agent_scale": 1.0 * SCALE,
                "agent_speed": 10.0 * SCALE,
                "arene-size": 5,
                "allowed-commands": 4,
                "command-count": 5,
                "command-duration": 3,
                "command-delay": 1,
                "use-command-alternative": False,
                "explosion-duration": 4,
                "explosion-delay": 4,
                "reward-command-failuer": -0.1,
                "reward-command-success": 0.1,
                "reward-episode-success": 0.0
            }

    def process_reset_params(reset_params):
        cloned_params = MortarMayhemEnv.default_reset_parameters.copy()
        if reset_params is not None:
            for k, v in reset_params.items():
                assert k in cloned_params.keys(), "Provided reset parameter (" + str(k) + ") is not valid. Check spelling."
                cloned_params[k] = v
        return cloned_params

    def __init__(self, headless = True) -> None:
        if headless:
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            pygame.display.set_caption("Environment")

        self.screen_dim = int(336 * SCALE)

        # Setup observation and action space
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space= spaces.Box(
                    low = 0.0,
                    high = 1.0,
                    shape = [self.screen_dim, self.screen_dim, 3],
                    dtype = np.float32)

        # Init PyGame screen
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim), pygame.NOFRAME)

        # Init debug window
        self.debug_window = None
        
        if headless:
            pygame.event.set_allowed(None)
        self.clock = pygame.time.Clock()

        self.rotated_agent_surface, self.rotated_agent_rect = None, None

    def _draw_surfaces(self, surfaces):
        # Draw all surfaces
        for surface in surfaces:
            if surface[0] is not None:
                self.screen.blit(surface[0], surface[1])
        pygame.display.flip()

    def reset(self, seed = None, return_info = True, options = None):
        super().reset(seed=seed)
        self.reset_params = MortarMayhemEnv.process_reset_params(options)

        # Track all rewards during one episode
        self.episode_rewards = []

        # Setup agent
        self.agent = CharacterController(self.screen_dim, self.reset_params["agent_speed"], self.reset_params["agent_scale"])
        spawn_pos = (42, 42)
        self.agent.rect.center = spawn_pos

        # Draw
        self.command = Command("up", SCALE, (0, 0))
        self.bg = pygame.Surface((self.screen_dim, self.screen_dim))
        self.bg.fill(0)
        self.tile = MortarTile(SCALE, (20, 20))
        self._draw_surfaces([(self.bg, (0, 0)), (self.tile.surface, (20, 20)), (self.agent.surface, self.agent.rect), (self.command.surface, (0, 0))])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs

    def step(self, action):
        # Move the agent's controlled character
        self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step(action)

        reward = 0
        done = False

        # Track all rewards
        self.episode_rewards.append(reward)

        if done:
            info = {
                "reward": sum(self.episode_rewards),
                "length": len(self.episode_rewards)
            }
        else:
            info = {}

        # Draw
        self._draw_surfaces([(self.bg, (0, 0)), (self.tile.surface, (20, 20)), (self.agent.surface, self.agent.rect), (self.command.surface, (0, 0))])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, reward, done, info

    def render(self, mode = "rgb_array"):
        if mode == "rgb_array":
            self.clock.tick(MortarMayhemEnv.metadata["render_fps"])
            return pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)


    def close(self):
            if self.debug_window is not None:
                self.debug_window.destroy()
            pygame.quit()

def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, help="The to be used seed for the environment's random number generator.", default=0)
    options = parser.parse_args()

    env = MortarMayhemEnv(headless = False)
    reset_params = {}
    vis_obs = env.reset(seed = options.seed, options = reset_params)
    img = env.render(mode = "rgb_array")
    done = False

    while not done:
        actions = [0, 0]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            actions[1] = 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            actions[0] = 2
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            actions[1] = 2
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            actions[0] = 1
        vis_obs, reward, done, info = env.step(actions)
        img = env.render(mode = "rgb_array")

        # Process event-loop
        for event in pygame.event.get():
        # Quit
            if event.type == pygame.QUIT:
                done = True

    print("episode reward: " + str(info["reward"]))
    print("episode length: " + str(info["length"]))

    env.close()
    exit()

if __name__ == "__main__":
    main()
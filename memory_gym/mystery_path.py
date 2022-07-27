import gym
import numpy as np
import os
import pygame

from argparse import ArgumentParser
from gym import  spaces
from memory_gym.pygame_assets import CharacterController, MysteryPath
from pygame._sdl2 import Window, Texture, Renderer

SCALE = 1.0

class MysteryPathEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "debug_rgb_array"],
        "render_fps": 25,
    }

    default_reset_parameters = {
                "agent_scale": 1.0 * SCALE,
                "agent_speed": 10.0 * SCALE,
                "arena_size": 5,
                "allowed_commands": 4,
                "command_count": 5,
                "command_show_duration": 3,
                "command_show_delay": 1,
                "explosion_duration": 6,
                "explosion_delay": 18,
                "reward_command_failure": -0.1,
                "reward_command_success": 0.1,
                "reward_episode_success": 0.0
            }

    def process_reset_params(reset_params):
        cloned_params = MysteryPathEnv.default_reset_parameters.copy()
        if reset_params is not None:
            for k, v in reset_params.items():
                assert k in cloned_params.keys(), "Provided reset parameter (" + str(k) + ") is not valid. Check spelling."
                cloned_params[k] = v
        assert cloned_params["allowed_commands"] >= 4 and cloned_params["allowed_commands"] <= 9
        assert cloned_params["arena_size"] >= 2 and cloned_params["arena_size"] <= 6
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

    def _build_debug_surface(self):
        surface = pygame.Surface((336 * SCALE, 336 * SCALE))
        return pygame.transform.scale(surface, (336, 336))

    def reset(self, seed = None, return_info = True, options = None):
        super().reset(seed=seed)

        # Check reset parameters for completeness and errors
        self.reset_params = MysteryPathEnv.process_reset_params(options)

        # Track all rewards during one episode
        self.episode_rewards = []

        # Setup path
        # Determine the start position on the screen's extent
        choice = self.np_random.choice([0, 1, 2, 3])
        if choice == 0:
            start = (0, self.np_random.integers(0, 7))
            end = (6, self.np_random.integers(0, 7))
        elif choice == 1:
            start = (6, self.np_random.integers(0, 7))
            end = (0, self.np_random.integers(0, 7))
        elif choice == 2:
            start = (self.np_random.integers(0, 7), 0)
            end = (self.np_random.integers(0, 7), 6)
        else:
            start = (self.np_random.integers(0, 7), 6)
            end = (self.np_random.integers(0, 7), 0)
        path = MysteryPath(7, 7, start, end, self.np_random)
        self.path_surface = pygame.Surface((self.screen_dim, self.screen_dim))
        self.path_surface.fill(0)
        path.draw_to_surface(self.path_surface, self.screen_dim // 7)


        # Setup the agent and sample its position
        self.agent = CharacterController(self.screen_dim, self.reset_params["agent_speed"], self.reset_params["agent_scale"])
        self.agent.rect.center = (start[0] * self.screen_dim // 7 + self.agent.radius, start[1] * self.screen_dim // 7 + self.agent.radius)

        # Draw
        self._draw_surfaces([(self.path_surface, (0, 0)), (self.agent.surface, self.agent.rect)])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs

    def step(self, action):
        reward = 0
        done = False

        # Move the agent's controlled character
        self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step(action, self.screen.get_rect())

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
        self._draw_surfaces([(self.path_surface, (0, 0)), (self.rotated_agent_surface, self.rotated_agent_rect)])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, reward, done, info

    def render(self, mode = "rgb_array"):
        if mode == "rgb_array":
            self.clock.tick(MysteryPathEnv.metadata["render_fps"])
            return np.fliplr(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8), 3)) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

    def close(self):
        if self.debug_window is not None:
            self.debug_window.destroy()
        pygame.quit()

def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, help="The to be used seed for the environment's random number generator.", default=0)
    options = parser.parse_args()

    env = MysteryPathEnv(headless = False)
    reset_params = {}
    seed = options.seed
    vis_obs = env.reset(seed = seed, options = reset_params)
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
        if keys[pygame.K_PAGEDOWN] or keys[pygame.K_PAGEUP]:
            if keys[pygame.K_PAGEUP]:
                seed += 1
            if keys[pygame.K_PAGEDOWN]:
                seed -= 1
            vis_obs = env.reset(seed = seed, options = reset_params)
            img = env.render(mode = "rgb_array")
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
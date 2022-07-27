import gym
import numpy as np
import os
import pygame

from argparse import ArgumentParser
from gym import  spaces
from memory_gym.pygame_assets import CharacterController, MysteryPath
from pygame._sdl2 import Window, Texture, Renderer

SCALE = 0.25

class MysteryPathEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "debug_rgb_array"],
        "render_fps": 25,
    }

    default_reset_parameters = {
                "agent_scale": 1.0 * SCALE,
                "agent_speed": 10.0 * SCALE,
                "cardinal_origin_choice": [0, 1, 2, 3],
                "show_origin": True,
                "show_goal": True,
                "reward_goal": 1.0,
                "reward_fall_off": 0.0,
                "reward_path_progress": 0.0
            }

    def process_reset_params(reset_params):
        cloned_params = MysteryPathEnv.default_reset_parameters.copy()
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

        # Init PyGame screen
        pygame.init()
        self.screen_dim = int(336 * SCALE)
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim), pygame.NOFRAME)
        self.clock = pygame.time.Clock()
        if headless:
            pygame.event.set_allowed(None)

        # Init debug window
        self.debug_window = None

        # Setup observation and action space
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space= spaces.Box(
                    low = 0.0,
                    high = 1.0,
                    shape = [self.screen_dim, self.screen_dim, 3],
                    dtype = np.float32)
        
        # Environment members
        self.rotated_agent_surface, self.rotated_agent_rect = None, None
        self.grid_dim = 7
        self.tile_dim = self.screen_dim / self.grid_dim

    def _draw_surfaces(self, surfaces):
        # Draw all surfaces
        for surface in surfaces:
            if surface[0] is not None:
                self.screen.blit(surface[0], surface[1])
        pygame.display.flip()

    def _build_debug_surface(self):
        surface = pygame.Surface((336 * SCALE, 336 * SCALE))
        surface.fill(0)
        self.mystery_path.draw_to_surface(surface, self.tile_dim, True, True, True, True)
        if self.rotated_agent_surface:
            surface.blit(self.rotated_agent_surface, self.rotated_agent_rect)
        else:
            surface.blit(self.agent.surface, self.agent.rect)
        return pygame.transform.scale(surface, (336, 336))

    def _normalize_agent_position(self, agent_position):
            return (agent_position[0] // self.tile_dim, agent_position[1] // self.tile_dim)

    def reset(self, seed = None, return_info = True, options = None):
        super().reset(seed=seed)
        self.current_seed = seed

        # Check reset parameters for completeness and errors
        self.reset_params = MysteryPathEnv.process_reset_params(options)

        # Track all rewards during one episode
        self.episode_rewards = []

        # Setup path
        # Determine the start and end position on the screen's extent
        cardinal_origin = self.np_random.choice(self.reset_params["cardinal_origin_choice"])
        if cardinal_origin == 0:
            self.start = (0, self.np_random.integers(0, self.grid_dim))
            self.end = (self.grid_dim - 1, self.np_random.integers(0, self.grid_dim))
        elif cardinal_origin == 1:
            self.start = (self.grid_dim - 1, self.np_random.integers(0, self.grid_dim))
            self.end = (0, self.np_random.integers(0, self.grid_dim))
        elif cardinal_origin == 2:
            self.start = (self.np_random.integers(0, self.grid_dim), 0)
            self.end = (self.np_random.integers(0, self.grid_dim), self.grid_dim - 1)
        else:
            self.start = (self.np_random.integers(0, self.grid_dim), self.grid_dim - 1)
            self.end = (self.np_random.integers(0, self.grid_dim), 0)
        
        # Procedurally generate the mystery path using A*
        self.mystery_path = MysteryPath(self.grid_dim, self.grid_dim, self.start, self.end, self.np_random)
        self.path_surface = pygame.Surface((self.screen_dim, self.screen_dim))
        self.path_surface.fill(0)
        self.mystery_path.draw_to_surface(self.path_surface, self.tile_dim, self.reset_params["show_origin"], self.reset_params["show_goal"])

        # Setup the agent and sample its position
        rotation = self.np_random.choice([0, 45, 90, 135, 180, 225, 270, 315])
        self.agent = CharacterController(self.screen_dim, self.reset_params["agent_speed"], self.reset_params["agent_scale"], rotation)
        # Place the agent on the path's starting position
        self.agent.rect.center = (self.start[0] * self.tile_dim + self.agent.radius, self.start[1] * self.tile_dim + self.agent.radius)
        self.normalized_agent_position = self._normalize_agent_position(self.agent.rect.center)

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

        # Check whether the agent reached the goal
        self.normalized_agent_position = self._normalize_agent_position(self.agent.rect.center)
        if self.normalized_agent_position == self.end:
            reward += self.reset_params["reward_goal"]
            done = True
        else:
            # Check whether the agent fell off the path
            on_path = False
            for node in self.mystery_path.path:
                if self.normalized_agent_position == (node.x, node.y):
                    on_path = True
                    break
            if not on_path:
                self.agent.rect.center = (self.start[0] * self.tile_dim + self.agent.radius, self.start[1] * self.tile_dim + self.agent.radius)
                reward += self.reset_params["reward_fall_off"]

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
        elif mode == "debug_rgb_array":
            # Create debug window if it doesn't exist yet
            if self.debug_window is None:
                self.debug_window = Window(size = (336, 336))
                self.debug_window.show()
                self.renderer = Renderer(self.debug_window)
            
            self.debug_window.title = "seed " + str(self.current_seed)
            self.clock.tick(MysteryPathEnv.metadata["render_fps"])

            debug_surface = self._build_debug_surface()
            texture = Texture.from_surface(self.renderer, debug_surface)
            texture.draw(dstrect=(0, 0))
            self.renderer.present()
            return np.fliplr(np.rot90(pygame.surfarray.array3d(self.renderer.to_surface()).astype(np.uint8), 3))

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
    img = env.render(mode = "debug_rgb_array")
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
                if not seed <= 0:
                    seed -= 1
            vis_obs = env.reset(seed = seed, options = reset_params)
            img = env.render(mode = "debug_rgb_array")
        vis_obs, reward, done, info = env.step(actions)
        img = env.render(mode = "debug_rgb_array")

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
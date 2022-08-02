import gym
import numpy as np
import os
import pygame

from argparse import ArgumentParser
from gym import  spaces
from memory_gym.character_controller import GridCharacterController
from memory_gym.pygame_assets import MysteryPath
from pygame._sdl2 import Window, Texture, Renderer

SCALE = 0.25

class GridMysteryPathEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "debug_rgb_array"],
        "render_fps": 3,
    }

    default_reset_parameters = {
                "agent_scale": 1.0 * SCALE,
                "cardinal_origin_choice": [0, 1, 2, 3],
                "show_origin": True,
                "show_goal": True,
                "reward_goal": 1.0,
                "reward_fall_off": 0.0,
                "reward_path_progress": 0.0
            }

    def process_reset_params(reset_params):
        cloned_params = GridMysteryPathEnv.default_reset_parameters.copy()
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
        self.action_space = spaces.Discrete(4)
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
        self.reset_params = GridMysteryPathEnv.process_reset_params(options)

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
        rotation = self.np_random.choice([0, 90, 180, 270])
        # Place the agent on the path's starting position
        start_pos = (self.start[0] * self.tile_dim + int(25 * SCALE), self.start[1] * self.tile_dim + int(25 * SCALE)) # TODO depend on agent radius and not hard coded int(25 * SCALE)
        self.normalized_agent_position = self._normalize_agent_position(start_pos)
        self.agent = GridCharacterController(SCALE, self.normalized_agent_position, self.mystery_path.to_grid(self.tile_dim), rotation)
        self.num_fails = 0

        # Draw
        self._draw_surfaces([(self.path_surface, (0, 0)), (self.agent.surface, self.agent.rect)])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs

    def step(self, action):
        reward = 0
        done = False
        success = 0

        # Move the agent's controlled character
        self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step(action)

        # Check whether the agent reached the goal
        self.normalized_agent_position = self._normalize_agent_position(self.agent.rect.center)
        if self.normalized_agent_position == self.end:
            reward += self.reset_params["reward_goal"]
            done = True
            success = 1
        else:
            # Check whether the agent fell off the path
            on_path = False
            for node in self.mystery_path.path:
                if self.normalized_agent_position == (node.x, node.y):
                    on_path = True
                    if not node.visited and not (node.x, node.y) == self.start and not (node.x, node.y) == self.end:
                        # Reward the agent for reaching a tile that it has not visisted before
                        reward += self.reset_params["reward_path_progress"]
                        node.visited = True
                    break
            if not on_path:
                # self.agent.rect.center = (self.start[0] * self.tile_dim + self.agent.radius, self.start[1] * self.tile_dim + self.agent.radius)
                start_pos = (self.start[0] * self.tile_dim + int(25 * SCALE), self.start[1] * self.tile_dim + int(25 * SCALE))
                self.agent.reset_position(self._normalize_agent_position(start_pos))
                reward += self.reset_params["reward_fall_off"]
                self.num_fails += 1

        # Track all rewards
        self.episode_rewards.append(reward)

        if done:
            info = {
                "reward": sum(self.episode_rewards),
                "length": len(self.episode_rewards),
                "success": success,
                "num_fails": self.num_fails,
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
            self.clock.tick(GridMysteryPathEnv.metadata["render_fps"])
            return np.fliplr(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8), 3)) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)
        elif mode == "debug_rgb_array":
            # Create debug window if it doesn't exist yet
            if self.debug_window is None:
                self.debug_window = Window(size = (336, 336))
                self.debug_window.show()
                self.renderer = Renderer(self.debug_window)
            
            self.debug_window.title = "seed " + str(self.current_seed)
            self.clock.tick(GridMysteryPathEnv.metadata["render_fps"])

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

    env = GridMysteryPathEnv(headless = False)
    reset_params = {}
    seed = options.seed
    vis_obs = env.reset(seed = seed, options = reset_params)
    img = env.render(mode = "debug_rgb_array")
    done = False

    while not done:
        actions = [0]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            actions[0] = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            actions[0] = 2
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
    print("success: " + str(bool(info["success"])))
    print("num fails: " + str(info["num_fails"]))

    env.close()
    exit()

if __name__ == "__main__":
    main()
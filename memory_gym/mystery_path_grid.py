import gymnasium as gym
import numpy as np
import os
import pygame

from argparse import ArgumentParser
from gymnasium import spaces
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
                "max_steps": 128,
                "agent_scale": 1.0 * SCALE,
                "cardinal_origin_choice": [0, 1, 2, 3],
                "show_origin": True,
                "show_goal": True,
                "visual_feedback": True,
                "reward_goal": 1.0,
                "reward_fall_off": 0.0,
                "reward_path_progress": 0.0,
                "reward_step": 0.0
            }

    def process_reset_params(reset_params):
        """Compares the provided reset parameters to the default ones. It asserts whether false reset parameters were provided.
        Missing reset parameters are filled with the default ones.

        Arguments:
            reset_params {dict} -- Provided reset parameters that are to be validated and completed

        Returns:
            dict -- Returns a complete and valid dictionary comprising the to be used reset parameters.
        """
        cloned_params = GridMysteryPathEnv.default_reset_parameters.copy()
        if reset_params is not None:
            for k, v in reset_params.items():
                assert k in cloned_params.keys(), "Provided reset parameter (" + str(k) + ") is not valid. Check spelling."
                cloned_params[k] = v
        return cloned_params

    def __init__(self, render_mode = None) -> None:
        self.render_mode = render_mode
        if render_mode is None:
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            pygame.display.set_caption("Environment")

        # Init PyGame screen
        pygame.init()
        self.screen_dim = int(336 * SCALE)
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim), pygame.NOFRAME)
        self.clock = pygame.time.Clock()
        if render_mode is None:
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
        surface.blit(self.fall_off_surface, self.fall_off_rect)
        return pygame.transform.scale(surface, (336, 336))

    def _normalize_agent_position(self, agent_position):
            return (agent_position[0] // self.tile_dim, agent_position[1] // self.tile_dim)

    def reset(self, seed = None, return_info = True, options = None):
        super().reset(seed=seed)
        self.current_seed = seed
        self.t = 0

        # Check reset parameters for completeness and errors
        self.reset_params = GridMysteryPathEnv.process_reset_params(options)
        self.max_episode_steps = self.reset_params["max_steps"]

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

        # Fall off surface to indicate that the agent lost the path
        dim = 40 * SCALE
        self.fall_off_surface = pygame.Surface((dim, dim))
        self.fall_off_rect = self.fall_off_surface.get_rect()
        self.fall_off_surface.fill(0)
        self.fall_off_surface.set_colorkey(0)
        pygame.draw.line(self.fall_off_surface, (255, 0, 0), (0, 0), (dim - 1, dim - 1), int(12 * SCALE))
        pygame.draw.line(self.fall_off_surface, (255, 0, 0), (dim - 1, 0), (0, dim - 1), int(12 * SCALE))
        self.fall_off_surface.set_alpha(0)

        # Setup the agent and sample its position
        rotation = 0 #self.np_random.choice([0, 90, 180, 270])
        # Place the agent on the path's starting position
        start_pos = (self.start[0] * self.tile_dim + int(25 * SCALE), self.start[1] * self.tile_dim + int(25 * SCALE)) # TODO depend on agent radius and not hard coded int(25 * SCALE)
        self.normalized_agent_position = self._normalize_agent_position(start_pos)
        self.agent = GridCharacterController(SCALE, self.normalized_agent_position, self.mystery_path.to_grid(self.tile_dim), rotation)
        self.rotated_agent_surface, self.rotated_agent_rect = self.agent.get_rotated_sprite(rotation)
        self.is_off_path = False
        self.num_fails = 0

        # Draw
        self._draw_surfaces([(self.path_surface, (0, 0)), (self.rotated_agent_surface, self.rotated_agent_rect)])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, {}

    def step(self, action):
        reward = 0
        done = False
        success = 0

        # Move the agent's controlled character
        if not self.is_off_path:
            self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step(action)
        else:
            # self.agent.rect.center = (self.start[0] * self.tile_dim + self.agent.radius, self.start[1] * self.tile_dim + self.agent.radius)
            start_pos = (self.start[0] * self.tile_dim + int(25 * SCALE), self.start[1] * self.tile_dim + int(25 * SCALE))
            self.agent.reset_position(self._normalize_agent_position(start_pos))
            self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step(0)

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
                reward += self.reset_params["reward_fall_off"]
                self.num_fails += 1
                if self.reset_params["visual_feedback"]:
                    self.fall_off_surface.set_alpha(255)
                self.is_off_path = True
            else:
                self.fall_off_surface.set_alpha(0)
                self.is_off_path = False
            self.fall_off_rect.center = self.rotated_agent_rect.center

        reward += self.reset_params["reward_step"]

        # Track all rewards
        self.episode_rewards.append(reward)

        # Time limit
        self.t += 1
        if self.t == self.max_episode_steps:
            done = True

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
        self._draw_surfaces([(self.path_surface, (0, 0)), (self.rotated_agent_surface, self.rotated_agent_rect), (self.fall_off_surface, self.fall_off_rect)])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, reward, done, False, info

    def render(self):
        if self.render_mode is not None:
            if self.render_mode == "rgb_array":
                self.clock.tick(GridMysteryPathEnv.metadata["render_fps"])
                return np.fliplr(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8), 3)) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)
            elif self.render_mode == "debug_rgb_array":
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

    env = GridMysteryPathEnv(render_mode = "debug_rgb_array")
    reset_params = {}
    seed = options.seed
    vis_obs, reset_info = env.reset(seed = seed, options = reset_params)
    img = env.render()
    done = False

    while not done:
        # Process event-loop
        keys = None
        for event in pygame.event.get():
            # Process key presses
            if event.type == pygame.KEYDOWN:
                keys = event.key
            # Quit
            if event.type == pygame.QUIT:
                done = True
        action = -1
        if keys == None:
            continue
        if keys == pygame.K_UP or keys == pygame.K_w:
            action = 3
        if keys == pygame.K_RIGHT or keys == pygame.K_d:
            action = 2
        if keys == pygame.K_LEFT or keys == pygame.K_a:
            action = 1
        if keys == pygame.K_PAGEDOWN or keys == pygame.K_PAGEUP:
            if keys == pygame.K_PAGEUP:
                seed += 1
            if keys == pygame.K_PAGEDOWN:
                if not seed <= 0:
                    seed -= 1
            vis_obs, reset_info = env.reset(seed = seed, options = reset_params)
            img = env.render()
        if action >= 0:
            vis_obs, reward, done, truncation, info = env.step(action)
            img = env.render()

    print("episode reward: " + str(info["reward"]))
    print("episode length: " + str(info["length"]))
    print("success: " + str(bool(info["success"])))
    print("num fails: " + str(info["num_fails"]))

    env.close()
    exit()

if __name__ == "__main__":
    main()
import gymnasium as gym
import numpy as np
import os
import pygame

from argparse import ArgumentParser
from gymnasium import  spaces
from memory_gym.environment import CustomEnv
from memory_gym.character_controller import CharacterController
from memory_gym.pygame_assets import Coin, GridPositionSampler, Spotlight, get_tiled_background_surface
from pygame._sdl2 import Window, Texture, Renderer

SCALE = 0.25

class EndlessSearingSpotlightsEnv(CustomEnv):
    metadata = {
        "render_modes": ["rgb_array", "debug_rgb_array"],
        "render_fps": 25,
    }

    default_reset_parameters = {
                # Spotlight parameters
                "max_steps": 1024,
                "steps_per_coin": 160,
                "initial_spawns": 3,
                "spawn_interval": 50,
                "spot_min_radius": 30.0 * SCALE,
                "spot_max_radius": 55.0 * SCALE,
                "spot_min_speed": 0.0025,
                "spot_max_speed": 0.0075,
                "spot_damage": 1.0,
                "visual_feedback": True,
                "black_background": False,
                "hide_chessboard": False,
                # Light Parameters
                "light_dim_off_duration": 6,
                "light_threshold": 255,
                # Coin Parameters
                "coin_scale": 1.5 * SCALE,
                "coin_show_duration": 6,
                "coins_visible": False,
                # Agent Parameters
                "agent_speed": 12.0 * SCALE,
                "agent_health": 20,
                "agent_scale": 1.0 * SCALE,
                "agent_visible": False,
                "sample_agent_position": True,
                "show_last_action": True,
                "show_last_positive_reward": True,
                # Reward Function
                "reward_inside_spotlight": 0.0,
                "reward_outside_spotlight": 0.0,
                "reward_death": 0.0,
                "reward_max_steps": 0.0,
                "reward_coin": 0.25,
            }

    def process_reset_params(reset_params):
        """Compares the provided reset parameters to the default ones. It asserts whether false reset parameters were provided.
        Missing reset parameters are filled with the default ones.

        Arguments:
            reset_params {dict} -- Provided reset parameters that are to be validated and completed

        Returns:
            dict -- Returns a complete and valid dictionary comprising the to be used reset parameters.
        """
        cloned_params = EndlessSearingSpotlightsEnv.default_reset_parameters.copy()
        if reset_params is not None:
            for k, v in reset_params.items():
                assert k in cloned_params.keys(), "Provided reset parameter (" + str(k) + ") is not valid. Check spelling."
                cloned_params[k] = v
        return cloned_params

    def __init__(self, render_mode = None) -> None:
        super().__init__()
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
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space= spaces.Box(
                    low = 0.0,
                    high = 1.0,
                    shape = [self.screen_dim, self.screen_dim, 3],
                    dtype = np.float32)
        
        # Optional information that is part of the returned info dictionary during reset and step
        # The absolute position (ground truth) of the agent is distributed using the info dictionary.
        self.has_ground_truth_info = True
        self.ground_truth_space = spaces.Box(
                    low = np.zeros((2), dtype=np.float32),
                    high = np.ones((2), dtype=np.float32),
                    shape = (4, ),
                    dtype = np.float32)

        # Environment members
        # Tiled background surface
        self.blue_background_surface = get_tiled_background_surface(self.screen, self.screen_dim, (0, 0, 255), SCALE)
        self.red_background_surface = get_tiled_background_surface(self.screen, self.screen_dim, (255, 0, 0), SCALE)

        # Spotlight surface
        self.spotlight_surface = pygame.Surface((self.screen_dim, self.screen_dim))
        self.spotlight_surface.fill(0)
        self.spotlight_surface.set_colorkey((255, 0, 0))

        # Agent boundaries
        self.walkable_rect = pygame.Rect(0, 16 * SCALE, self.screen_dim, self.screen_dim - 16 * SCALE)

        # Init grid spawner
        self.grid_sampler = GridPositionSampler(self.screen_dim)
        # self.grid_sampler = GridPositionSampler(self.screen_dim - 16 * SCALE, self.screen_dim // 24)

        self.rotated_agent_surface, self.rotated_agent_rect = None, None

    def _draw_surfaces(self, surfaces):
        # Draw all surfaces
        for surface in surfaces:
            if surface[0] is not None:
                self.screen.blit(surface[0], surface[1])
        pygame.display.flip()

    def _build_debug_surface(self):
        surface = pygame.Surface((336 * SCALE, 336 * SCALE))
        # Create coin surface
        coin_surface = pygame.Surface((self.screen_dim, self.screen_dim))
        coin_surface.fill(255)
        coin_surface.set_colorkey(255)
        self.coin.draw(coin_surface)

        # Gather surfaces
        surfs = [(self.bg, (0, 0)), (self.spotlight_surface, (0, 0)), (coin_surface, (0, 0))]
        # Retrieve the rotated agent surface or the original one
        if self.rotated_agent_surface is not None:
            surfs.append((self.rotated_agent_surface, self.rotated_agent_rect))
        else:
            surfs.append(self.agent.get_rotated_sprite(0))
        surfs.append((self.top_bar_surface, (0, 0)))
        # Blit all surfaces
        for surf, rect in surfs:
            if surf is not None:
                surface.blit(surf, rect)

        return pygame.transform.scale(surface, (336, 336))

    def _step_spotlight_task(self):
        reward = 0.0
        done = False
        # Spawn spotlights
        self.spawn_timer += 1
        if self.spawn_timer >= self.reset_params["spawn_interval"]:
            self.spotlights.append(Spotlight(self.screen_dim, self.np_random.integers(self.reset_params["spot_min_radius"], self.reset_params["spot_max_radius"] + 1),
                                                        self.np_random.uniform(self.reset_params["spot_min_speed"], self.reset_params["spot_max_speed"]), self.np_random, 
                                                        self.reset_params["black_background"]))
            self.spawn_timer = 0

        # Draw spotlights and check whether the agent is visible or not
        self.spotlight_surface.fill(0)
        spotlight_hit = 0
        for spot in self.spotlights:        
            # Remove spotlights that finished traversal
            if spot.done:
                self.spotlights.remove(spot)
            # Draw spotlight and measure distance to agent
            else:
                spot.draw(self.spotlight_surface)
                if spot.is_agent_inside(self.agent):
                    spotlight_hit += 1

        if spotlight_hit > 0:
            self.current_agent_health -= self.reset_params["spot_damage"]
            reward += self.reset_params["reward_inside_spotlight"]
            width = int(self.screen_dim // 2 * (1 - self.current_agent_health / self.agent_health))
            pygame.draw.rect(self.top_bar_surface, (255, 0, 0), (0, 0, width, 16 * SCALE))
            # Render the background tiles in red if visual feedback is desired
            if self.reset_params["visual_feedback"]:
                bg = self.red_background_surface
            else:
                bg = self.blue_background_surface
        else:
            bg = self.blue_background_surface
            reward += self.reset_params["reward_outside_spotlight"]

        if self.reset_params["black_background"]:
            bg.fill(0)

        # Determine done
        if self.current_agent_health <= 0:
            done = True
            reward += self.reset_params["reward_death"]

        return reward, done, bg

    def _process_spawn_pos(self, spawn_pos, offset = 30):
        x = spawn_pos[0]
        y = spawn_pos[1]
        offset = int(offset * SCALE)
        if x < offset:
            x = offset
        elif x > self.screen_dim - offset:
            x = self.screen_dim - offset
        if y < offset:
            y = offset
        elif y > self.screen_dim - offset:
            y = self.screen_dim - offset
        return (x, y)

    def _spawn_coin(self):
        self.coin_surface = pygame.Surface((self.screen_dim, self.screen_dim))
        self.coin_surface.fill(255)
        self.coin_surface.set_colorkey(255)
        # Reset grid sampler
        self.grid_sampler.reset(self.np_random)
        # Block current agent position
        if self.coin is not None:
            self.grid_sampler.block_spawn_position((self.coin.location[0], self.coin.location[1]), 28)
        spawn_pos = self.grid_sampler.sample(28)
        spawn_pos = (spawn_pos[0] + self.np_random.integers(2, 4), spawn_pos[1] + self.np_random.integers(2, 4))
        spawn_pos = self._process_spawn_pos(spawn_pos)
        self.coin = Coin(self.reset_params["coin_scale"], spawn_pos)
        self.coin.draw(self.coin_surface)

    def _step_coin_task(self):
        reward = 0.0
        # Check whether the agent collected a coin and redraw the coin surface
        update_coin_surface = False
        if self.coin.is_agent_inside(self.agent):
            reward += self.reset_params["reward_coin"]
            update_coin_surface = True
            self.coins_collected += 1
            self.steps_between_coins.append(self.t)
            self.t = 0
            # Spawn new coin
            self._spawn_coin()
        # Redraw coins if at least one was collected
        if update_coin_surface:
            self.coin_surface.fill(255)
            self.coin.draw(self.coin_surface)
        return reward

    def reset(self, seed = None, return_info = True, options = None):
        super().reset(seed=seed)
        self.current_seed = seed
        self.reset_params = EndlessSearingSpotlightsEnv.process_reset_params(options)
        self.max_episode_steps = self.reset_params["max_steps"]
        self.t = 0
        self.steps_between_coins = []

        if self.reset_params["hide_chessboard"]:
            self.blue_background_surface.fill((255, 255, 255))
            self.red_background_surface.fill((255, 255, 255))

        # Reset spawner
        self.grid_sampler.reset(self.np_random)

        # Track all rewards during one episode
        self.episode_rewards = []

        # Setup agent
        self.last_action = [0, 0]   # The agent shall sense its last action to potentially infer its postion from its past actions
        rotation = self.np_random.choice([0, 45, 90, 135, 180, 225, 270, 315])
        self.agent = CharacterController(self.reset_params["agent_speed"], self.reset_params["agent_scale"], rotation)
        if self.reset_params["sample_agent_position"]:
            spawn_pos = self.grid_sampler.sample(28)
            spawn_pos = (spawn_pos[0] + self.np_random.integers(2, 4), spawn_pos[1] + self.np_random.integers(2, 4))
        else:
            spawn_pos = (self.screen_dim // 2, self.screen_dim // 2)
            self.grid_sampler.block_spawn_position(spawn_pos)
        self.agent.rect.center = spawn_pos
        self.agent_health = self.reset_params["agent_health"]
        self.current_agent_health = self.agent_health
        # The agent's interface, which is a bar on the top of the screen
        # Render the agent's health as a green bar on the first half of the screen width
        self.quarter_width = int(self.screen_dim // 4)
        self.top_bar_surface = pygame.Surface((self.screen_dim, 16 * SCALE))
        pygame.draw.rect(self.top_bar_surface, (0, 255, 0), (0, 0, self.quarter_width * 2, 16 * SCALE))
        # Render the last action of the agent
        if self.reset_params["show_last_action"]:
            self.action_colors = [(120, 120, 120), (116, 1, 113), (255, 94, 14)]
            self.act_rect_0 =  (self.quarter_width * 2, 0, self.quarter_width, 16 * SCALE)
            self.act_rect_1 =  (self.quarter_width * 3, 0, self.quarter_width, 16 * SCALE)
            pygame.draw.rect(self.top_bar_surface, self.action_colors[0], self.act_rect_0)
            pygame.draw.rect(self.top_bar_surface, self.action_colors[0], self.act_rect_1)
        if self.reset_params["show_last_positive_reward"]:
            self.last_reward = 0.0
            if self.reset_params["show_last_action"]:
                self.coin_bar_rect = (int(self.quarter_width * 2.75), 0, int(self.quarter_width * 0.5), 16 * SCALE)
            else:
                self.coin_bar_rect = (int(self.quarter_width * 2), 0, int(self.quarter_width * 2), 16 * SCALE)

        # Setup spotlights
        if self.reset_params["light_dim_off_duration"] > 0:
            self.spotlight_surface.set_alpha(0)
        else:
            self.spotlight_surface.set_alpha(self.reset_params["light_threshold"])
        self.spotlights = []
        self.spawn_timer = 0
        for _ in range(self.reset_params["initial_spawns"]):
            self.spotlights.append(Spotlight(self.screen_dim, self.np_random.integers(self.reset_params["spot_min_radius"], self.reset_params["spot_max_radius"] + 1),
                                                            self.np_random.uniform(self.reset_params["spot_min_speed"], self.reset_params["spot_max_speed"]), self.np_random,
                                                            self.reset_params["black_background"]))

        # Spawn coin
        self.coins_collected = 0
        self.coin = None
        self._spawn_coin()

        # Draw initially all surfaces
        self.bg = self.blue_background_surface
        if self.reset_params["black_background"]:
            self.bg.fill(0)
        surfaces = [(self.bg, (0, 0)), (self.spotlight_surface, (0, 0)), (self.top_bar_surface, (0, 0))]
        spot_surface_id = 1
        # Coin surface
        if self.reset_params["coins_visible"] or self.t < self.reset_params["coin_show_duration"]:
            surfaces.insert(spot_surface_id + 1, (self.coin_surface, (0, 0)))
        else:
            surfaces.insert(spot_surface_id, (self.coin_surface, (0, 0)))
            spot_surface_id += 1
        # Agent surface
        if self.reset_params["agent_visible"]:
            surfaces.insert(spot_surface_id + 3, (self.agent.get_rotated_sprite(0)))
        else:
            surfaces.insert(spot_surface_id, (self.agent.get_rotated_sprite(0)))
            spot_surface_id += 1
        self._draw_surfaces(surfaces)

        # Show spawn mask for debugging purposes
        # import matplotlib.pyplot as plt
        # plt.imshow(np.flip(np.rot90(self.grid_sampler.spawn_mask), axis=0))
        # plt.show()

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        # Return the visual observation and the ground truth
        return vis_obs, {"ground_truth": np.asarray([float(self.agent.rect.center) / float(self.screen_dim), float(self.coin.rect.center) / float(self.screen_dim)])}

    def step(self, action):
        # Move the agent's controlled character
        self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step(action, self.walkable_rect)

        # Render the last action of the agent
        pygame.draw.rect(self.top_bar_surface, self.action_colors[self.last_action[0]], self.act_rect_0)
        pygame.draw.rect(self.top_bar_surface, self.action_colors[self.last_action[1]], self.act_rect_1)
        self.last_action = action

        # Dim light untill off
        if self.spotlight_surface.get_alpha() <= self.reset_params["light_threshold"]:
            if self.reset_params["light_dim_off_duration"] > 0:
                self.spotlight_surface.set_alpha(self.spotlight_surface.get_alpha() + int(255 / self.reset_params["light_dim_off_duration"]))
            else:
                self.spotlight_surface.set_alpha(self.reset_params["light_threshold"])

        # Process tasks
        # Spotlight task
        reward = 0.0
        r, spotlights_done, self.bg = self._step_spotlight_task()
        reward += r
        # Coin collection task
        r = self._step_coin_task()
        reward += r

        # Determine done
        done = False
        if spotlights_done:
            done = True

        # Time limit
        self.t += 1
        if self.t == self.reset_params["steps_per_coin"]:
            done = True

        # Render the last reward of the agent
        if self.reset_params["show_last_positive_reward"]:
            if self.last_reward > 0:
                pygame.draw.rect(self.top_bar_surface, (255, 255, 0), self.coin_bar_rect)
            else:
                pygame.draw.rect(self.top_bar_surface, (50, 50, 50), self.coin_bar_rect)
            self.last_reward = reward

        # Draw all surfaces
        surfaces = [(self.bg, (0, 0)), (self.spotlight_surface, (0, 0)), (self.top_bar_surface, (0, 0))]
        spot_surface_id = 1
        # Coin surface
        if self.reset_params["coins_visible"] or self.t < self.reset_params["coin_show_duration"]:
            surfaces.insert(spot_surface_id + 1, (self.coin_surface, (0, 0)))
        else:
            surfaces.insert(spot_surface_id, (self.coin_surface, (0, 0)))
            spot_surface_id += 1
        # Agent surface
        if self.reset_params["agent_visible"]:
            surfaces.insert(spot_surface_id + 3, (self.rotated_agent_surface, self.rotated_agent_rect))
        else:
            surfaces.insert(spot_surface_id, (self.rotated_agent_surface, self.rotated_agent_rect))
            spot_surface_id += 1
        self._draw_surfaces(surfaces)

        # Track all rewards
        self.episode_rewards.append(reward)

        if done:
            info = {
                "reward": sum(self.episode_rewards),
                "length": len(self.episode_rewards),
                "agent_health": self.current_agent_health / self.agent_health,
                "coins_collected": self.coins_collected,
                "ground_truth": np.asarray([float(self.agent.rect.center) / float(self.screen_dim), float(self.coin.rect.center) / float(self.screen_dim)]),
                # "mean_steps_between_coins": sum(self.steps_between_coins) / self.coins_collected
            }
        else:
            # Ground truth: agent and coin position
            info = {"ground_truth": np.asarray([float(self.agent.rect.center) / float(self.screen_dim), float(self.coin.rect.center) / float(self.screen_dim)])}

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, reward, done, False, info

    def close(self):
        if self.debug_window is not None:
            self.debug_window.destroy()
        pygame.quit()

    def render(self):
        if self.render_mode is not None:
            if self.render_mode == "rgb_array":
                self.clock.tick(EndlessSearingSpotlightsEnv.metadata["render_fps"])
                return np.fliplr(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8), 3)) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)
            elif self.render_mode == "debug_rgb_array":
                # Create debug window if it doesn't exist yet
                if self.debug_window is None:
                    self.debug_window = Window(size = (336, 336))
                    self.debug_window.show()
                    self.renderer = Renderer(self.debug_window)

                self.debug_window.title = "seed " + str(self.current_seed)
                self.clock.tick(EndlessSearingSpotlightsEnv.metadata["render_fps"])

                debug_surface = self._build_debug_surface()
                texture = Texture.from_surface(self.renderer, debug_surface)
                texture.draw(dstrect=(0, 0))
                self.renderer.present()
                return np.fliplr(np.rot90(pygame.surfarray.array3d(self.renderer.to_surface()).astype(np.uint8), 3))

def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, help="The to be used seed for the environment's random number generator.", default=0)
    options = parser.parse_args()

    env = EndlessSearingSpotlightsEnv(render_mode = "debug_rgb_array")
    reset_params = {}
    seed = options.seed
    vis_obs, reset_info = env.reset(seed = options.seed, options = reset_params)
    img = env.render()
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
            vis_obs, reset_info = env.reset(seed = seed, options = reset_params)
            img = env.render()
        vis_obs, reward, done, truncation, info = env.step(actions)
        img = env.render()

        # Process event-loop
        for event in pygame.event.get():
        # Quit
            if event.type == pygame.QUIT:
                done = True

    print("episode reward: " + str(info["reward"]))
    print("episode length: " + str(info["length"]))
    print("agent health: " + str(info["agent_health"]))
    print("coins collected: " + str(info["coins_collected"]))
    # print("mean steps between coins: " + str(info["mean_steps_between_coins"]))

    env.close()
    exit()

if __name__ == "__main__":
    main()
from distutils.spawn import spawn
import gym
import math
import numpy as np
import os
import pygame

from gym import  spaces
from memory_gym.pygame_assets import CharacterController, Coin, Exit, GridPositionSampler, Spotlight, get_tiled_background_surface

SCALE = 0.25

class SearingSpotlightsEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 25,
    }

    default_reset_parameters = {
                # Spotlight parameters
                "initial_spawns": 4,
                "num_spawns": 30,
                "initial_spawn_interval": 30,
                "spawn_interval_threshold": 10,
                "spawn_interval_decay": 0.95,
                "spot_min_radius": 30.0 * SCALE,
                "spot_max_radius": 55.0 * SCALE,
                "spot_min_speed": 0.0025,
                "spot_max_speed": 0.0075,
                "spot_damage": 1.0,
                # Light Parameters
                "light_dim_off_duration": 10,
                # Coin Parameters
                "min_num_coins": 2,
                "max_num_coins": 5,
                "coin_scale": 1.5 * SCALE,
                # Exit Parameters
                "use_exit": True,
                "exit_scale": 2.0 * SCALE,
                # Agent Parameters
                "agent_speed": 10.0 * SCALE,
                "agent_health": 100,
                "agent_scale": 1.0 * SCALE,
                # Reward Function
                "reward_inside_spotlight": -0.01,
                "reward_outside_spotlight": 0.0,
                "reward_exit": 1.0,
                "reward_max_steps": 0.0,
                "reward_coin": 0.0,
            }

    def process_reset_params(reset_params):
        cloned_params = SearingSpotlightsEnv.default_reset_parameters.copy()
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
                    low = 0,
                    high = 255,
                    shape = [self.screen_dim, self.screen_dim, 3],
                    dtype = np.uint8)

        # Init PyGame screen
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim), pygame.NOFRAME)
        if headless:
            pygame.event.set_allowed(None)
        self.clock = pygame.time.Clock()

        # Initialize surfaces
        # Tiled background surface
        self.blue_background_surface = get_tiled_background_surface(self.screen, self.screen_dim, (0, 0, 255), SCALE)
        self.red_background_surface = get_tiled_background_surface(self.screen, self.screen_dim, (255, 0, 0), SCALE)

        # Spotlight surface
        self.spotlight_surface = pygame.Surface((self.screen_dim, self.screen_dim))
        self.spotlight_surface.set_colorkey((255, 255, 255))

        # Init grid spawner
        self.grid_sampler = GridPositionSampler(self.screen_dim, self.screen_dim // 24)

    def _compute_spawn_intervals(self, reset_params) -> list:
        intervals = []
        initial = reset_params["initial_spawn_interval"]
        for i in range(reset_params["num_spawns"]):
            intervals.append(int(initial + reset_params["spawn_interval_threshold"]))
            initial = initial * math.pow(reset_params["spawn_interval_decay"], 1)
        return intervals

    def _draw_surfaces(self, surfaces):
        # Draw all surfaces
        for surface in surfaces:
            if surface[0] is not None:
                self.screen.blit(surface[0], surface[1])
        pygame.display.flip()

    def _step_spotlight_task(self):
        reward = 0.0
        done = False
        # Spawn spotlights
        self.spawn_timer += 1
        if self.spawn_intervals:
            if self.spawn_timer >= self.spawn_intervals[0]:
                self.spotlights.append(Spotlight(self.screen_dim, self.np_random.integers(self.reset_params["spot_min_radius"], self.reset_params["spot_max_radius"] + 1),
                                                            self.np_random.uniform(self.reset_params["spot_min_speed"], self.reset_params["spot_max_speed"]), self.np_random))
                self.spawn_intervals.pop()
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
            bg = self.red_background_surface
            self.agent_health -= self.reset_params["spot_damage"]
            reward += self.reset_params["reward_inside_spotlight"]
        else:
            bg = self.blue_background_surface
            reward += self.reset_params["reward_outside_spotlight"]

        # Determine done
        if self.agent_health <= 0:
            done = True

        return reward, done, bg

    def _spawn_coins(self):
        self.coin_surface = pygame.Surface((self.screen_dim, self.screen_dim))
        self.coin_surface.fill(255)
        self.coin_surface.set_colorkey(255)
        for _ in range(self.np_random.integers(self.reset_params["min_num_coins"], self.reset_params["max_num_coins"] + 1)):
            spawn_pos = self.grid_sampler.sample(2)
            spawn_pos = (spawn_pos[0] + self.np_random.integers(2, 4), spawn_pos[1] + self.np_random.integers(2, 4))
            coin = Coin(self.reset_params["coin_scale"], spawn_pos)
            coin.draw(self.coin_surface)
            self.coins.append(coin)

    def _spawn_exit(self):
        spawn_pos = self.grid_sampler.sample(3)
        spawn_pos = (spawn_pos[0] + self.np_random.integers(2, 4), spawn_pos[1] + self.np_random.integers(2, 4))
        self.exit = Exit(spawn_pos, self.reset_params["exit_scale"])
        self.exit.draw(open=False)

    def _step_coin_task(self):
        reward = 0.0
        done = False
        # Check whether the agent collected a coin and redraw the coin surface
        update_coin_surface = False
        for coin in self.coins:
            if coin.is_agent_inside(self.agent):
                self.coins.remove(coin)
                reward += self.reset_params["reward_coin"]
                update_coin_surface = True
        # Redraw coins if at least one was collected
        if update_coin_surface:
            self.coin_surface.fill(255)
            for coin in self.coins:
                coin.draw(self.coin_surface)
        if not self.coins:
            done = True
        return reward, done

    def _step_exit_task(self, coins_done):
        reward = 0.0
        done = False
        if coins_done:
            self.exit.draw(open = True)
            if self.exit.is_agent_inside(self.agent):
                done = True
                reward = self.reset_params["reward_exit"]
        return reward, done

    def reset(self, seed = None, return_info = True, options = None):
        super().reset(seed=seed)
        self.reset_params = SearingSpotlightsEnv.process_reset_params(options)

        # Reset spawner
        self.grid_sampler.reset(self.np_random)

        # Track all rewards during one episode
        self.episode_rewards = []

        # Setup agent
        self.agent = CharacterController(self.screen_dim, self.reset_params["agent_speed"], self.reset_params["agent_scale"])
        spawn_pos = self.grid_sampler.sample(5)
        spawn_pos = (spawn_pos[0] + self.np_random.integers(2, 4), spawn_pos[1] + self.np_random.integers(2, 4))
        self.agent.rect.center = spawn_pos
        self.agent_health = self.reset_params["agent_health"]

        # Setup spotlights
        self.spawn_intervals = self._compute_spawn_intervals(self.reset_params)
        self.spotlight_surface.set_alpha(0)
        self.spotlights = []
        self.spawn_timer = self.spawn_intervals[0] # ensure that the first spotlight is spawned right away
        for _ in range(self.reset_params["initial_spawns"]):
            self.spotlights.append(Spotlight(self.screen_dim, self.np_random.integers(self.reset_params["spot_min_radius"], self.reset_params["spot_max_radius"] + 1),
                                                            self.np_random.uniform(self.reset_params["spot_min_speed"], self.reset_params["spot_max_speed"]), self.np_random))

        # Spawn coin and exit entities if applicable
        self.coins = []
        if self.reset_params["min_num_coins"] > 0:
            self._spawn_coins()
        else:
            self.coin_surface = None
        if self.reset_params["use_exit"]:
            self._spawn_exit()
        else:
            self.exit_surface = None

        # Draw initially all surfaces
        self._draw_surfaces([(self.blue_background_surface, (0, 0)), (self.coin_surface, (0, 0)), (self.exit.surface, self.exit.rect),
                            (self.agent.surface, self.agent.rect), (self.spotlight_surface, (0, 0))])

        # Show spawn mask for debugging purposes
        # import matplotlib.pyplot as plt
        # plt.imshow(np.flip(np.rot90(self.grid_sampler.spawn_mask), axis=0))
        # plt.show()

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs

    def step(self, action):
        # Move the agent's controlled character
        agent_surface, agent_rect = self.agent.step(action)

        # Dim light untill off
        if self.spotlight_surface.get_alpha() <= 255:
            self.spotlight_surface.set_alpha(self.spotlight_surface.get_alpha() + int(255 / self.reset_params["light_dim_off_duration"]))

        # Process tasks
        # Spotlight task
        reward = 0.0
        r, spotlights_done, bg = self._step_spotlight_task()
        reward += r
        # Coin collection task
        if self.reset_params["min_num_coins"] > 0:
            r, coins_done = self._step_coin_task()
            reward += r
        else:
            coins_done = True
        # Exit task
        if self.reset_params["use_exit"]:
            r, exit_done = self._step_exit_task(coins_done)
            reward += r
        else:
            exit_done = False

        # Determine done
        done = False
        if spotlights_done:
            done = True
        elif coins_done:
            if self.reset_params["use_exit"]:
                if exit_done:
                    done = True
                else:
                    done = False
            else:
                if self.reset_params["min_num_coins"] > 0:
                    done = True

        # Draw all surfaces
        self._draw_surfaces([(bg, (0, 0)), (self.coin_surface, (0, 0)), (self.exit.surface, self.exit.rect),
                            (agent_surface, agent_rect), (self.spotlight_surface, (0, 0))])

        # Track all rewards
        self.episode_rewards.append(reward)

        if done:
            info = {
                "reward": sum(self.episode_rewards),
                "length": len(self.episode_rewards)
            }
        else:
            info = {}

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, reward, done, info

    def close(self):
        pygame.quit()

    def render(self, mode = "rgb_array"):
        self.clock.tick(SearingSpotlightsEnv.metadata["render_fps"])
        return pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

def main():
    env = SearingSpotlightsEnv(headless = False)
    reset_params = {}
    vis_obs = env.reset(options = reset_params)

    done = False

    while not done:
        img = env.render()

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
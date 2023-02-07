import gymnasium as gym
import numpy as np
import os
import pygame

from argparse import ArgumentParser
from gymnasium import  spaces
from gymnasium.utils import seeding
from memory_gym.character_controller import GridCharacterController
from memory_gym.mortar_mayhem_grid import GridMortarMayhemEnv
from memory_gym.pygame_assets import Command, MortarArena, calc_max_episode_steps

SCALE = 0.25

class GridMortarMayhemTaskBEnv(GridMortarMayhemEnv):
    metadata = {
        "render_modes": ["rgb_array", "debug_rgb_array"],
        "render_fps": 1,
    }

    default_reset_parameters = {
                "agent_scale": 1.0 * SCALE,
                "arena_size": 5,
                "allowed_commands": 5,
                "command_count": [10],
                "explosion_duration": [2],
                "explosion_delay": [6],
                "visual_feedback": True,
                "reward_command_failure": 0.0,
                "reward_command_success": 0.1,
                "reward_episode_success": 0.0
            }

    def process_reset_params(reset_params):
        """Compares the provided reset parameters to the default ones. It asserts whether false reset parameters were provided.
        Missing reset parameters are filled with the default ones.

        Arguments:
            reset_params {dict} -- Provided reset parameters that are to be validated and completed

        Returns:
            dict -- Returns a complete and valid dictionary comprising the to be used reset parameters.
        """
        cloned_params = GridMortarMayhemTaskBEnv.default_reset_parameters.copy()
        if reset_params is not None:
            for k, v in reset_params.items():
                assert k in cloned_params.keys(), "Provided reset parameter (" + str(k) + ") is not valid. Check spelling."
                cloned_params[k] = v
        assert cloned_params["allowed_commands"] >= 4 and cloned_params["allowed_commands"] <= 9
        assert cloned_params["arena_size"] >= 2 and cloned_params["arena_size"] <= 6
        assert max(cloned_params["command_count"]) <= 20, "20 commands are allowed at maximum"
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
        self.max_num_commands = 20
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "visual_observation": spaces.Box(
                    low = 0.0,
                    high = 1.0,
                    shape = [self.screen_dim, self.screen_dim, 3],
                    dtype = np.float32),
                "vector_observation": spaces.Box(
                    low = np.zeros((self.max_num_commands * 9), dtype=np.float32),
                    high = np.ones((self.max_num_commands * 9), dtype=np.float32),
                    shape = (self.max_num_commands * 9, ),
                    dtype = np.float32)
            }
        )

        # Environment members
        self.rotated_agent_surface, self.rotated_agent_rect = None, None

    def _encode_commands_one_hot(self, commands):
        one_hot_commands = np.zeros((self.max_num_commands * 9), dtype = np.float32)
        for c, command in enumerate(commands):
            if command == "stay":
                one_hot_commands[9 * c + 0] = 1.0
            elif command == "right":
                one_hot_commands[9 * c + 1] = 1.0
            elif command == "left":
                one_hot_commands[9 * c + 2] = 1.0
            elif command == "up":
                one_hot_commands[9 * c + 3] = 1.0
            elif command == "down":
                one_hot_commands[9 * c + 4] = 1.0
            elif command == "right_down":
                one_hot_commands[9 * c + 5] = 1.0
            elif command == "right_up":
                one_hot_commands[9 * c + 6] = 1.0
            elif command == "left_down":
                one_hot_commands[9 * c + 7] = 1.0
            elif command == "left_up":
                one_hot_commands[9 * c + 8] = 1.0
        return one_hot_commands

    def reset(self, seed = None, return_info = True, options = None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        self.current_seed = seed

        # Check reset parameters for completeness and errors
        self.reset_params = GridMortarMayhemTaskBEnv.process_reset_params(options)
        self.max_episode_steps = calc_max_episode_steps(max(self.reset_params["command_count"]),
                                                            0,
                                                            0,
                                                            max(self.reset_params["explosion_delay"]),
                                                            max(self.reset_params["explosion_duration"]))

        # Track all rewards during one episode
        self.episode_rewards = []

        # Setup the arena and place it on the center of the screen
        self.bg = pygame.Surface((self.screen_dim, self.screen_dim))
        self.arena = MortarArena(SCALE, self.reset_params["arena_size"])
        self.arena.rect.center = (self.screen_dim // 2, self.screen_dim // 2)

        # Setup the agent and sample its position
        spawn_pos = self.arena.get_tile_global_position(self.np_random.integers(0, self.reset_params["arena_size"] ** 2))
        translate_x = self.arena.rect.center[0] - self.arena.local_center[0] + self.arena.tile_dim // 2
        translate_y = self.arena.rect.center[1] - self.arena.local_center[1] + self.arena.tile_dim // 2
        agent_pos = spawn_pos[0] + translate_x, spawn_pos[1] + translate_y
        self.normalized_agent_position = self._normalize_agent_position(agent_pos)
        self.agent = GridCharacterController(SCALE, self.normalized_agent_position, self.arena.to_grid())

        # Sample the entire command sequence
        self._commands = self._generate_commands(self.normalized_agent_position)
        # Prepare list which prepares all steps (i.e. frames) for the visualization
        self._command_visualization = None

        # Init episode members
        self._target_pos = (self.normalized_agent_position[0] + Command.COMMANDS[self._commands[0]][0],
                            self.normalized_agent_position[1] + Command.COMMANDS[self._commands[0]][1])
        self._current_command = 0       # the current to be executed command
        self._command_steps = 0         # the current step while executing a command (i.e. death tiles off)
        self._command_verify_step = 0   # the current step while the command is being evaluated (i.e. death tiles on)
        # Sample execution delay and duration
        self._explosion_duration = self.np_random.choice(self.reset_params["explosion_duration"])
        self._explosion_delay = self.np_random.choice(self.reset_params["explosion_delay"])

        # Draw
        self._draw_surfaces([(self.bg, (0, 0)), (self.arena.surface, self.arena.rect), (self.agent.surface, self.agent.rect)])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)
        # Retrieve the encoded commands for the observation space
        self._commands_one_hot = self._encode_commands_one_hot(self._commands)

        return {"visual_observation": vis_obs, "vector_observation": self._commands_one_hot}, {}

    def step(self, action):
        reward = 0
        done = False
        success = 0
        command = None

        # All commands are fully observable, the agent can move now, while the command execution logic is running
        # Move the agent's controlled character
        self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step(action)
        self.normalized_agent_position = self._normalize_agent_position(self.rotated_agent_rect.center)

        # Process the command execution logic
        # One command is alive for explosion delay steps
        verify = self._command_steps % self._explosion_delay == 0 and self._command_steps > 0

        # Run the verification logic on whether the agent succeeded on moving to the target tile
        if verify and not self.arena.tiles_on:
            if self._current_command < self.num_commands:
                self._current_command += 1

                # Turn on the death tiles
                self.arena.toggle_tiles(self._target_pos, self.reset_params["visual_feedback"])

                # Check if the agent is on the target position
                if self.normalized_agent_position == self._target_pos:
                    # Success!
                    reward += self.reset_params["reward_command_success"]
                # If the agent is not on the target position, terminate the episode
                else:
                    # Failure!
                    done = True
                    reward += self.reset_params["reward_command_failure"]
            # Finish the episode once all commands are completed
            if self._current_command >= self.num_commands:
                # All commands completed!
                done = True
                success = 1
                reward += self.reset_params["reward_episode_success"]
            self._command_steps = 1

        # Keep the death tiles on for as long as the explosion duration
        if self.arena.tiles_on:
            if self._command_verify_step % self._explosion_duration == 0 and self._command_verify_step > 0:
                # Turn death tiles off
                self.arena.toggle_tiles(None, self.reset_params["visual_feedback"])
                self._command_verify_step = 0
                if self._current_command < self.num_commands:
                    # Update target position
                    self._target_pos = (self._target_pos[0] + Command.COMMANDS[self._commands[self._current_command]][0],
                                        self._target_pos[1] + Command.COMMANDS[self._commands[self._current_command]][1])
            else:
                # The agent dies upon walking on a death tile
                if not self.normalized_agent_position == self._target_pos:
                    # Failure!
                    done = True
                    reward = self.reset_params["reward_command_failure"]
                self._command_verify_step += 1
        else:
            self._command_steps +=1

        # Track all rewards
        self.episode_rewards.append(reward)

        if done:
            info = {
                "reward": sum(self.episode_rewards),
                "length": len(self.episode_rewards),
                "success": success,
                "commands_completed": (self._current_command - 1 + success) / self.num_commands,
            }
        else:
            info = {}
        
        # Draw
        surfaces = [(self.bg, (0, 0)), (self.arena.surface, self.arena.rect), (self.rotated_agent_surface, self.rotated_agent_rect)]
        if command is not None:
            surfaces.append((command.surface, ((self.screen_dim // 2) - command.rect_dim // 2, (self.screen_dim // 2) - command.rect_dim // 2)))
        self._draw_surfaces(surfaces)

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return {"visual_observation": vis_obs, "vector_observation": self._commands_one_hot}, reward, done, False, info

def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, help="The to be used seed for the environment's random number generator.", default=0)
    options = parser.parse_args()

    env = GridMortarMayhemTaskBEnv(render_mode = "debug_rgb_array")
    reset_params = {}
    seed = options.seed
    obs, reset_info = env.reset(seed = options.seed, options = reset_params)
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
        actions = [-1]
        if keys == None:
            continue
        if keys == pygame.K_UP or keys == pygame.K_w:
            actions[0] = 3
        if keys == pygame.K_RIGHT or keys == pygame.K_d:
            actions[0] = 2
        if keys == pygame.K_LEFT or keys == pygame.K_a:
            actions[0] = 1
        if keys == pygame.K_SPACE:
            actions[0] = 0
        if keys == pygame.K_PAGEDOWN or keys == pygame.K_PAGEUP:
            if keys == pygame.K_PAGEUP:
                seed += 1
            if keys == pygame.K_PAGEDOWN:
                if not seed <= 0:
                    seed -= 1
            obs, reset_info = env.reset(seed = seed, options = reset_params)
            img = env.render()    
        if actions[0] >= 0:
            obs, reward, done, truncation, info = env.step(actions)
            img = env.render()

    print("episode reward: " + str(info["reward"]))
    print("episode length: " + str(info["length"]))
    print("success: " + str(bool(info["success"])))
    print("commands completed: " + str(info["commands_completed"]))

    env.close()
    exit()

if __name__ == "__main__":
    main()
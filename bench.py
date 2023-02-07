import memory_gym
import gymnasium as gym
import time
import numpy as np

env = gym.make("SearingSpotlights-v0")
# env = gym.make("MortarMayhem-v0")
# env = gym.make("MysteryPath-v0")
# env = gym.make("MortarMayhemB-Grid-v0")
# env = gym.make("MysteryPath-Grid-v0")

def run():
    fps_list = []
    rew_list = []

    for _ in range(1000):
        steps = 0
        start_time = time.time()
        obs, reset_info = env.reset()
        done = False
        while not done:
            obs, reward, done, truncation, info = env.step(env.action_space.sample())
            steps += 1
        fps = steps / (time.time() - start_time)
        fps_list.append(fps)
        rew_list.append(info["exit_success"])
    fps_list = np.array(fps_list)
    print("Mean steps per second: " + str(np.mean(fps_list)) + " std: " + str(np.std(fps_list)))
    rew_list = np.array(rew_list)
    print("Mean reward: " + str(np.mean(rew_list)) + " std: " + str(np.std(rew_list)))

# import cProfile, pstats
# profiler = cProfile.Profile()
# profiler.enable()
run()
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats()

env.close()
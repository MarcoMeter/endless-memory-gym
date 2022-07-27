import memory_gym
import gym
import time
import numpy as np

env = gym.make("SearingSpotlights-v0")
# env = gym.make("MortarMayhem-v0")
# env = gym.make("MysteryPath-v0")

def run():
    fps_list = []

    for _ in range(10):
        steps = 0
        start_time = time.time()
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, _ = env.step([0, 0])
            steps += 1
        fps = steps / (time.time() - start_time)
        fps_list.append(fps)
    fps_list = np.array(fps_list)
    print("Mean steps per second: " + str(np.mean(fps_list)) + " std: " + str(np.std(fps_list)))

# import cProfile, pstats
# profiler = cProfile.Profile()
# profiler.enable()
run()
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats()

env.close()
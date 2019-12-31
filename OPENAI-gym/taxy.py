import gym
import numpy as np

env = gym.make("Taxi-v2")
observation=env.reset()

# visualizacion valor devuelto al resetear retorno
print(observation)

# mostrar estado actual entorno
env.render()

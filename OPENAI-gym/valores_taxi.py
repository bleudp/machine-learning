import gym
import numpy as np
""" test only """
env = gym.make("Taxi-v2")
observation = env.reset()
# Visualizamos el valor devuelto
print(observation)

# mostrar estado actual
env.render()

# movimientos del taxi
action = 1 # mover arriba
observation, reward, done, info = env.step(action)
print(env.step(action))
env.render()

action = 2 # mover derecha
observation, reward, done, info = env.step(action)
print(env.step(action))
env.render()

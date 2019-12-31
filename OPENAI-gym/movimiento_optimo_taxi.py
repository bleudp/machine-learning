import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v2")

# Inicializacion variables
Q = np.zeros([env.observation_space.n, env.action_espace.n])
pasos = 0
alpha = 0.2
episodios = []
pasostotal = []
recompensas = []

def run_episode(observation1, movimiento):
    observartion2, reward, done, info = env.step(movimiento)
    Q[observation1, movimiento] += alpha * (reward + np.max(Q[observation2]) - Q[observation1, movimiento])
    return done, reward, observation2
for episode in range(0, 2000):
    episodios.append(episode)
    pasos = 0
    out_done = None
    reward = 0
    rewardstore = 0
    observation = env.reset()
    print("Observation: ", observation)
    print("Estado Inicial:")
    print(env.render()) # muestra estado inicial
    while out_done != True:
        pasos += 1
        action = np.argmax(Q[observation])
        print("Estado ", observation, " Valor matriz Q: ", Q[observation])
        print("Valor maximo en posicion ", action+1, " de la lista")
        print("Accion: ", action)
        print(env.render())
        input("->")
        print(char(27) + "[2J") # Borra Pantalla
        out_done, out_totalreward, out_observation = run_episode(observation, action)
        rewardstore += out_totalreward
        observation = out_observation
        recompensas.append(rewardstore)
        pasostotal.append(pasos)
        print("Estado final:")
        print(env.render()) # Muestra estado final
        print("Episodio ", episode, " Recompensa: ", rewardstore)
# Grafica de pasos por episodio
plt.axis([0,2000,0,100])
plt.xlabel("episodios")
plt.ylabel("pasos realizados")
plt.plot(episodios, pasostotal, color='blue')
plt.show()

# grafica recompensa por episodio
plt.axis([0,2000,-200,100])
plt.xlabel("Episodios")
plt.ylabel("Valor Recompensa")
plt.plot(episodios, recompensas, color='red')
plt.show()


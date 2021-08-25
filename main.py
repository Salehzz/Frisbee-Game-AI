import numpy
import gym
import random

n_iter = 10000
Lrate = 0.1
Drate = 0.99
exploration_rate = 1
max_e = 1
min_e = 0.01
decay = 0.001
table=0
env=gym.make("FrozenLake-v0")

def update_table(state,action,reward,new_state):
    global table
    table[state, action] = table[state, action] * (1 - Lrate) + Lrate * (reward + Drate * numpy.max(table[new_state, :]))

def update_exploration_rate(episode):
    global exploration_rate
    exploration_rate = min_e + (max_e - min_e) * numpy.exp(-decay * episode)

def examples():
    global table,env
    print("\n3 Examples :\n")
    for episode in range(3):
        state = env.reset()
        print("EPISODE : ", episode+1, "\n")
        done=False
        while(not done):
            env.render()
            action = numpy.argmax(table[state, :])
            new_state, reward, done, info = env.step(action)
            if done:
                env.render()
                if reward == 1:
                    print("=>Goal!\n\n\n\n")
                else:
                    print("=>Hole!\n\n\n\n")
                break
            state = new_state

def learning_rate():
    global table,env
    wins=0
    for episode in range(1000):
        state = env.reset()
        done=False
        while(not done):
            action = numpy.argmax(table[state, :])
            new_state, reward, done, info = env.step(action)
            state = new_state
        wins+=reward
    return(wins/1000)

def simulate_one():
    global exploration_rate,table,env
    done=False
    state=env.reset()
    while(not done):
        randome = random.uniform(0, 1)
        if randome > exploration_rate:
            action = numpy.argmax(table[state, :])
        else:
            action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        update_table(state,action,reward,new_state)
        state = new_state
    return reward

def simulate(kind):
    global exploration_rate,table,env
    env = gym.make("FrozenLake-v0", map_name=kind)
    action_size=env.action_space.n
    state_size=env.observation_space.n
    table = numpy.zeros((state_size, action_size))
    wins=0
    for episode in range(n_iter):
        wins+=simulate_one()
        update_exploration_rate(episode)
    print("learning rate for {} for first 10000runs : {}".format(kind,str((wins/n_iter))))
    print("\n\nTable :\n")
    print(table)
    print("\n\nWin ratio(learning rate for {} after learning for 1000runs) : {}".format(kind,learning_rate()))
    input("\n\nEnter for examples :")
    examples()
    env.close()

for kind in ["4x4","8x8"]:
    if(kind=="8x8"):
        input("Enter for 8*8")
        min_e=0.00000000
        decay=0.00000000
        exploration_rate = 1
    simulate(kind)

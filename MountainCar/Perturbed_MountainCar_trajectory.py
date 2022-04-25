import torch
import gym
import numpy as np
from DDPG.DPG_Actor_NN import Actor_NN
from ModifiedMountainCar_env import ModifiedInvMCar


# Test how chaotic the InvertedPendulm is, load a good policy, compute the necessary actions for a trajectory
# storing the states along the way, then freeze the action and perturb the first "n_perturbations" actions (i.e. 1) and compute
# another trajectory for those actions, assessing how a small pertub to the first n actions affects the final state output

seeds = 0
torch.manual_seed(seeds)
np.random.seed(seeds)
n_perturbations = 1

std_perturbation = 0.01
max_t_steps = 999

env = gym.make("MountainCarContinuous-v0")
env = ModifiedInvMCar(env,easier=False)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
max_action =1

# Initialise two policy, one being the correct one and the other to be perturbed slightly
pol_nn = Actor_NN(Input_size=state_size,Hidden_size = 64,max_action=max_action).double()

parameters = torch.load("/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/ContMCarPolicy.pt")
pol_nn.load_state_dict(parameters)



## Compute the actions and then freeze them, perturbing only the first n_perturbations actions

unpertubed_states = []
action_set = []
c_state = env.reset()
unpertubed_states.append(c_state)

done = False
# Compute a good set of actions
t = 0
while not done:

    det_action = pol_nn(torch.from_numpy(c_state)).detach()
    next_st, rwd, done, _ = env.step(det_action.numpy())
    c_state = next_st
    unpertubed_states.append(next_st)
    action_set.append(det_action)
    t +=1

# "Freeze action", perturb first n and compute a state trajectory

perturbed_states = []
perturbed_action_set = []
c_state = env.reset()
perturbed_states.append(c_state)


# "Freeze the actions" and slightly perturb the first one to see chaos
for e in range(t):

    # Perturb weights
    # if t >1:
    #     det_action = pol_nn(torch.from_numpy(c_state)).detach()
    #
    # else:
    #     det_action = perturb_pol_nn(torch.from_numpy(c_state)).detach()
    #     print(det_action)

    #Perturb actions:
    if e >=n_perturbations:
        det_action = action_set[e]
    else:
        torch.manual_seed(seeds)
        perturbation = torch.randn((action_size,)) *std_perturbation
        det_action = action_set[e] - perturbation





    next_st, rwd, done, _ = env.step(det_action.numpy())
    c_state = next_st
    perturbed_states.append(next_st)
    perturbed_action_set.append(det_action)


torch.save(unpertubed_states,"/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/MountainCar_ActionUnPerturbed_States_s"+str(seeds)+".pt")
torch.save(perturbed_states,"/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/MountainCar_ActionPerturbed_States_s"+str(seeds)+".pt")
torch.save(action_set,"/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/MountainCar_ActionUnPerturbed_Actions_s"+str(seeds)+".pt")
torch.save(perturbed_action_set,"/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/MountainCar_ActionPerturbed_Actions_s"+str(seeds)+".pt")
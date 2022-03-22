import torch
import gym
import numpy as np
from DDPG.DPG_Actor_NN import Actor_NN
from InvertedPendulum.ModifiedPendulum import ModifiedInvPendulum


# Test how chaotic the InvertedPendulm is, load a good policy, compute the necessary actions for a trajectory
# storing the states along the way, then freeze the action and perturb the first "n_perturbations" actions (i.e. 1) and compute
# another trajectory for those actions, assessing how a small pertub to the first n actions affects the final state output

seeds = 0
torch.manual_seed(seeds)
np.random.seed(seeds)
n_perturbations = 1

std_perturbation = 0.005
max_t_steps = 200

env = gym.make("Pendulum-v0")
env = ModifiedInvPendulum(env)

# Initialise two policy, one being the correct one and the other to be perturbed slightly
pol_nn = Actor_NN().double()
perturb_pol_nn = Actor_NN().double()

parameters = torch.load("/Users/px19783/PycharmProjects/ChaoticEnv/InvertedPendulum/Results/PendulmPolicy.pt")
pol_nn.load_state_dict(parameters)

## Perturb the parameters:
# perturb_params = {}
# for key, value in parameters.items():
#     perturb_params[key] = parameters[key].clone() + torch.randn(parameters[key].size())*std_perturbation
#
# perturb_pol_nn.load_state_dict(perturb_params)

## Check that the parameters have been perturbed:
# print(perturb_pol_nn.l1.weight)
# print(pol_nn.l1.weight,'\n')
# print(perturb_pol_nn .l1.bias)
# print(pol_nn.l1.bias)

## Compute the actions and then freeze them, perturbing only the first n_perturbations actions

unpertubed_states = []
action_set = []
c_state = env.reset()
unpertubed_states.append(c_state)

# Compute a good set of actions
for t in range(max_t_steps):

    det_action = pol_nn(torch.from_numpy(c_state)).detach()
    next_st, rwd, done, _ = env.step(det_action.numpy())
    c_state = next_st
    unpertubed_states.append(next_st)
    action_set.append(det_action)

# "Freeze action", perturb first n and compute a state trajectory

perturbed_states = []
perturbed_action_set = []
c_state = env.reset()
perturbed_states.append(c_state)

# "Freeze the actions" and slightly perturb the first one to see chaos
for t in range(max_t_steps):

    # Perturb weights
    # if t >1:
    #     det_action = pol_nn(torch.from_numpy(c_state)).detach()
    #
    # else:
    #     det_action = perturb_pol_nn(torch.from_numpy(c_state)).detach()
    #     print(det_action)

    #Perturb actions:
    if t >=n_perturbations:
        det_action = action_set[t]
    else:
        det_action = action_set[t] - torch.randn((1,)) *std_perturbation


    next_st, rwd, done, _ = env.step(det_action.numpy())
    c_state = next_st
    perturbed_states.append(next_st)
    perturbed_action_set.append(det_action)



torch.save(unpertubed_states,"/Users/px19783/PycharmProjects/ChaoticEnv/InvertedPendulum/Results/Pendulum_ActionUnPerturbed_States_s"+str(seeds)+".pt")
torch.save(perturbed_states,"/Users/px19783/PycharmProjects/ChaoticEnv/InvertedPendulum/Results/Pendulum_ActionPerturbed_States_s"+str(seeds)+".pt")
torch.save(action_set,"/Users/px19783/PycharmProjects/ChaoticEnv/InvertedPendulum/Results/Pendulum_ActionUnPerturbed_Actions_s"+str(seeds)+".pt")
torch.save(perturbed_action_set,"/Users/px19783/PycharmProjects/ChaoticEnv/InvertedPendulum/Results/Pendulum_ActionPerturbed_Actions_s"+str(seeds)+".pt")
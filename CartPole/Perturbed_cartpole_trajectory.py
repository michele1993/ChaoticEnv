import torch
import gym
import numpy as np
from CartPole.REINFORCE.Actor_network import Policy_net
from ModifiedCartPole import ModifiedCartPole

# Test how chaotic Cartpole is, load a good policy, perturb weight for the first action and see how trajectory changes
# with all other action not perturbed (using a modified version of cartpole which always start in the same state)

seeds = 0
torch.manual_seed(seeds)
np.random.seed(seeds)

std_perturbation = 0.1
env = gym.make("CartPole-v0")
env = ModifiedCartPole(env)

# Initialise two policy, one being the correct one and the other to be perturbed slightly
pol_nn = Policy_net(output_size = 2)
perturb_pol_nn = Policy_net(output_size = 2)

parameters = torch.load("/Users/px19783/PycharmProjects/ChaoticEnv/CartPole/Results/CartPolePolicy.pt")
pol_nn.load_state_dict(parameters)

# Perturb the parameters:
perturb_params = {}
for key, value in parameters.items():
    perturb_params[key] = parameters[key].clone() + torch.randn(parameters[key].size())*std_perturbation

perturb_pol_nn.load_state_dict(perturb_params)

# print(perturb_pol_nn.l1.weight)
# print(pol_nn.l1.weight,'\n')
# print(perturb_pol_nn .l1.bias)
# print(pol_nn.l1.bias)

state_trajectory = []
current_st = env.reset()

state_trajectory.append(current_st)
done = False
t = 1

while not done:

    current_st = torch.FloatTensor([current_st])
    if t >1:
        det_action = pol_nn(torch.tensor(current_st)).detach()
    else:
        det_action = perturb_pol_nn(torch.tensor(current_st)).detach()

    det_action = torch.argmax(det_action)
    next_st, _, done, _ = env.step(int(det_action.numpy()))
    current_st = next_st
    state_trajectory.append(next_st)

torch.save(state_trajectory,"/Users/px19783/PycharmProjects/ChaoticEnv/CartPole/Results/CartPolePerturbedStates_s"+str(seeds)+".pt")

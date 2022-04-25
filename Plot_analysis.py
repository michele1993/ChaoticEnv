import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# Extract data for Pendulum:
Pendulm_unperturbed_states = torch.load('/Users/px19783/PycharmProjects/ChaoticEnv/InvertedPendulum/Results/Pendulum_ActionUnPerturbed_States_s0.pt')
Pendulm_perturbed_states = torch.load('/Users/px19783/PycharmProjects/ChaoticEnv/InvertedPendulum/Results/Pendulum_ActionPerturbed_States_s0.pt')

Pendulm_unperturbed_actions = torch.load('/Users/px19783/PycharmProjects/ChaoticEnv/InvertedPendulum/Results/Pendulum_ActionUnPerturbed_Actions_s0.pt')
Pendulm_perturbed_actions = torch.load('/Users/px19783/PycharmProjects/ChaoticEnv/InvertedPendulum/Results/Pendulum_ActionPerturbed_Actions_s0.pt')

Pendulm_gradient = pd.read_pickle('/Users/px19783/PycharmProjects/ChaoticEnv/InvertedPendulum/Results/Pendulum_gradient.pkl')

# Accuracy:
Pendulm_acc = []
for gamma in range(5,10):

    pd_data = pd.read_pickle('/Users/px19783/PycharmProjects/ChaoticEnv/InvertedPendulum/Results/p_gamma_reward0.'+str(gamma)+'.pkl')
    Pendulm_acc.append(pd_data.to_numpy()[:,1])


# convert panda data frame to numpy array for the gradient values
Pendulm_gradient = Pendulm_gradient.to_numpy()[:,3:]



# Compute squared difference between states
Pendulm_unperturbed_states  = np.array(Pendulm_unperturbed_states)
Pendulm_perturbed_states  = np.array(Pendulm_perturbed_states)

Pendulm_difference_states = np.abs(Pendulm_perturbed_states - Pendulm_unperturbed_states)
Pendulm_difference_states = np.cumsum(Pendulm_difference_states,axis=0)


# Compute squared difference between actions
Pendulm_unperturbed_actions  = np.array(Pendulm_unperturbed_actions)
Pendulm_perturbed_actions  = np.array(Pendulm_perturbed_actions)

Pendulm_difference_actions = np.abs(Pendulm_perturbed_actions - Pendulm_unperturbed_actions)

Pendulm_t = np.linspace(0,len(Pendulm_unperturbed_states),len(Pendulm_unperturbed_states))

# Extract data for MountainCar:
MC_unperturbed_states = torch.load('/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/MountainCar_ActionUnPerturbed_States_s0.pt')
MC_perturbed_states = torch.load('/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/MountainCar_ActionPerturbed_States_s0.pt')

MC_unperturbed_actions = torch.load('/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/MountainCar_ActionUnPerturbed_Actions_s0.pt')
MC_perturbed_actions = torch.load('/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/MountainCar_ActionPerturbed_Actions_s0.pt')

MC_gradient = pd.read_pickle('/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/MountainCar_gradient.pkl')

# Accuracy:
MC_acc = []
for gamma in range(5,10):

    pd_data = pd.read_pickle('/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/m_gamma_reward0.'+str(gamma)+'.pkl')
    MC_acc.append(pd_data.to_numpy()[:,1])


# convert panda data frame to numpy array for the gradient values
MC_gradient = MC_gradient.to_numpy()[:,3:]

# Compute squared difference between states
MC_unperturbed_states  = np.array(MC_unperturbed_states)
MC_perturbed_states  = np.array(MC_perturbed_states)

MC_difference_states = np.abs(MC_perturbed_states - MC_unperturbed_states)
MC_difference_states = np.cumsum(MC_difference_states,axis=0)

# Compute squared difference between actions
MC_unperturbed_actions  = np.array(MC_unperturbed_actions)
MC_perturbed_actions  = np.array(MC_perturbed_actions)

MC_difference_actions = np.abs(MC_perturbed_actions - MC_unperturbed_actions)

MC_t = np.linspace(0,len(MC_unperturbed_states),len(MC_unperturbed_states))

font_s = 7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
fig, axs = plt.subplots(nrows=2, ncols=4,figsize=(7, 3.5),
                                  gridspec_kw={'wspace': 0.5, 'hspace': 0.4, 'left': 0.1, 'right': 0.98, 'bottom': 0.1,
                                               'top': 0.94})

x_axis_label = 't-step'
col = ['tab:blue','tab:green','tab:orange']
lab= ["x-coordinate",'y-coordinate','velocity']
gamma_colors = ['black', 'blue','darkslateblue','mediumslateblue','darkviolet']
gamma_labels = ['$\gamma = 0.5$','$\gamma = 0.6$', '$\gamma = 0.7$', '$\gamma = 0.8$', '$\gamma = 0.9$']
#gradient_label = r'$\frac{d \sum_t r_t}{d a_0}}$'
gradient_label = 'MB-DPG gradient'

# Plot MountainCar: -----------------------------------
# Accuracy:
for gamma in range(5):
    acc = MC_acc[gamma]
    t = np.linspace(0, len(acc), len(acc))
    axs[0,0].plot(t,acc, color=gamma_colors[gamma], label=gamma_labels[gamma])

axs[0,0].spines['right'].set_visible(False)
axs[0,0].spines['top'].set_visible(False)
axs[0,0].set_ylabel('Accuracy')
axs[0,0].legend(loc='upper left',bbox_to_anchor=(0.35, 0.85),fontsize=font_s,frameon=False)
axs[0,0].set_ylabel('Accuracy',fontsize = font_s)
axs[0,0].set_title('Performance',fontsize=font_s)
#axs[1,0].set_ylim([-10000, -5000])
#axs[1,0].legend(loc='upper left',bbox_to_anchor=(0.25, 1.1),fontsize=6,frameon=False)

# Gradient:
for gamma in range(5):

    axs[0,1].plot(MC_t,MC_gradient[gamma][0],color=gamma_colors[gamma])

axs[0,1].spines['right'].set_visible(False)
axs[0,1].spines['top'].set_visible(False)
axs[0,1].set_ylim([0, 1000])
axs[0,1].set_ylabel(gradient_label)
axs[0,1].set_title('Exploding gradients',fontsize=font_s)

# State difference:
axs[0,2].plot(MC_t,MC_difference_states)
axs[0,2].spines['right'].set_visible(False)
axs[0,2].spines['top'].set_visible(False)
axs[0,2].set_ylabel('Cumulative state difference')
axs[0,2].set_title('Chaotic states',fontsize=font_s)



# Action difference:
axs[0,3].plot(MC_t[:-1],MC_difference_actions,color='gray')
axs[0,3].set_ylim([-0.2, 0.2])
axs[0,3].spines['right'].set_visible(False)
axs[0,3].spines['top'].set_visible(False)
axs[0,3].set_ylabel('Action difference')
axs[0,3].set_title('Action perturbation',fontsize=font_s)


# Plot Pendulum: -----------------------------------
# Accuracy:
for gamma in range(5):
    acc = Pendulm_acc[gamma]
    t = np.linspace(0, len(acc), len(acc))
    axs[1,0].plot(t,acc, color=gamma_colors[gamma], label=gamma_labels[gamma])

axs[1,0].spines['right'].set_visible(False)
axs[1,0].spines['top'].set_visible(False)
axs[1,0].set_ylim([-10000, -5000])
axs[1,0].set_xlabel('Episodes')
axs[1,0].set_ylabel('Accuracy',fontsize = font_s)


# Gradient:
for gamma in range(5):

    axs[1,1].plot(Pendulm_t,Pendulm_gradient[gamma][0][:201], color=gamma_colors[gamma], label=gamma_labels[gamma])

axs[1,1].spines['right'].set_visible(False)
axs[1,1].spines['top'].set_visible(False)
axs[1,1].set_ylim([0, 1000])
#axs[0,3].legend(loc='upper left',bbox_to_anchor=(0.25, 0.95),fontsize=font_s,frameon=False)
axs[1,1].set_ylabel(gradient_label)
axs[1,1].set_xlabel(x_axis_label)



# State difference:
for c in range(3):
    axs[1,2].plot(Pendulm_t,Pendulm_difference_states[:,c],label=lab[c], color=col[c])

axs[1,2].spines['right'].set_visible(False)
axs[1,2].spines['top'].set_visible(False)
axs[1,2].set_ylabel('Cumulative state difference')
axs[1,2].set_xlabel(x_axis_label)
axs[1,2].legend(loc='upper left',bbox_to_anchor=(0.01, 1),fontsize=font_s,frameon=False)


# Action difference:
axs[1,3].plot(Pendulm_t[:-1],Pendulm_difference_actions,color='gray')
axs[1,3].set_ylim([-0.2, 0.2])
axs[1,3].spines['right'].set_visible(False)
axs[1,3].spines['top'].set_visible(False)
axs[1,3].set_ylabel('Action difference')
axs[1,3].set_xlabel(x_axis_label)



#plt.show()
plt.savefig('/Users/px19783/Desktop/Chaos_diagram', format='png', dpi=800)
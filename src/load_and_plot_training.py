import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_dataset(exp_name, method_name, task_name, epoch):
    directory = Path(__file__).parent / 'dataset' / exp_name / method_name / task_name
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = str(epoch) + '.npz'
    path_name = directory / file_name
    
    data = np.load(path_name, allow_pickle=True)

    return data['arr_0'][()]

# task_reward_bounds = [(-100, -26), (-100, -62), (-100, -21), (-100, -19)]
# task_names = ['composite', 'sequential', 'OR', 'IF']
method_names = ['lof', 'fsa', 'greedy', 'flat', 'RM']
method_plot_names = ['LOF-VI', 'LOF-QL', 'Greedy', 'Flat Options', 'Reward Machines']
method_colors = ['b', 'r', 'g', 'y', 'c']
task_reward_bounds = [(-300, 0)]
task_names = ['or', 'if', 'sequential', 'composite']
# method_names = ['lof']
# method_plot_names = ['LOF-VI']
# method_colors = ['b']

h_env_horizon = 1000 # number of training steps per epoch
h_epochs = [i for i in range(0, 7501, 250)]
h_steps = np.array(h_epochs) * h_env_horizon
rm_epochs = [i for i in range(0, 10001, 250)]
rm_env_horizon = 1000
rm_steps = np.array(rm_epochs) * rm_env_horizon

num_exp = 2 # number of separate training runs

def get_plot_data_for_task(task_num, task_name, num_exp=2):
    method_max_rewards = []
    method_min_rewards = []
    method_ave_rewards = []

    for method_name in method_names:
        if method_name == 'RM':
            epochs = rm_epochs
        else:
            epochs = h_epochs
        num_data = len(epochs)
        max_rewards = [-np.inf]*num_data
        min_rewards = [np.inf]*num_data
        ave_rewards = [0]*num_data
        std_rewards = [0]*num_data
        for epoch_i, epoch in enumerate(epochs):
            first_data = load_dataset('lunchbox', method_name, task_name, epoch)
            num_trials = len(first_data['reward'])
            for i in range(num_exp):
                results = load_dataset('lunchbox', method_name, task_name, epoch)
                std_rewards[epoch_i] = np.std(np.array(results['reward']))
                for j in range(num_trials):
                    reward = results['reward'][j]
                    # for each experiment, average reward over the tasks
                    ave_rewards[epoch_i] += reward / (num_trials*num_exp)
                    if reward > max_rewards[epoch_i]:
                        max_rewards[epoch_i] = reward
                    if reward < min_rewards[epoch_i]:
                        min_rewards[epoch_i] = reward

        method_max_rewards.append(np.array(ave_rewards) + np.array(std_rewards))
        method_min_rewards.append(np.array(ave_rewards) - np.array(std_rewards))
        # method_max_rewards.append(max_rewards)
        # method_min_rewards.append(min_rewards)
        method_ave_rewards.append(ave_rewards)

    return method_ave_rewards, method_min_rewards, method_max_rewards


def get_plot_data_over_tasks():
    method_max_rewards = []
    method_min_rewards = []
    method_ave_rewards = []

    for method_name in method_names:
        if method_name == 'RM':
            epochs = rm_epochs
        else:
            epochs = h_epochs
        num_data = len(epochs)

        max_rewards = [-np.inf]*num_data
        min_rewards = [np.inf]*num_data
        ave_rewards = [0]*num_data
        std_rewards = [0]*num_data

        first_data = load_dataset('lunchbox', method_name, task_names[0], 0)
        num_trials = len(first_data['reward'])
        num_trials = 1
        for i in range(num_exp):
            task_ave_rewards_list = [[0 for nt in range(num_trials)] for ei in range(len(epochs))]
            # for each experiment, average reward over the tasks
            task_ave_rewards = [0]*num_data
            for k in range(num_trials):
                for j, task_name in enumerate(task_names):
                    for epoch_i, epoch in enumerate(epochs):
                        results = load_dataset('lunchbox', method_name, task_name, epoch)
                       
                        reward = results['reward'][k]
                        task_ave_rewards[epoch_i] += reward/(len(task_names)*num_trials)
                        task_ave_rewards_list[epoch_i][k] += reward/len(task_names)

                for ei in range(len(epochs)):
                    ave_reward = task_ave_rewards_list[ei][k]
                    if ave_reward > max_rewards[ei]:
                        max_rewards[ei] = ave_reward
                    if ave_reward < min_rewards[ei]:
                        min_rewards[ei] = ave_reward
                    
                
            for n, ave_reward in enumerate(task_ave_rewards):
                ave_rewards[n] += ave_reward / (num_exp)

            for epoch_i, epoch in enumerate(epochs):
                std_rewards[epoch_i] = np.std(task_ave_rewards_list[epoch_i])

        method_max_rewards.append(np.array(ave_rewards) + np.array(std_rewards))
        method_min_rewards.append(np.array(ave_rewards) - np.array(std_rewards))
        # method_max_rewards.append(max_rewards)
        # method_min_rewards.append(min_rewards)
        method_ave_rewards.append(ave_rewards)

    return method_ave_rewards, method_min_rewards, method_max_rewards

def plot_data_over_tasks():
    method_ave_rewards, method_min_rewards, \
        method_max_rewards = get_plot_data_over_tasks()


    for i, (method_name, method_color, ave_reward, min_reward, max_reward) in enumerate(zip(
            method_plot_names, method_colors, method_ave_rewards, method_min_rewards, method_max_rewards)):
        if 'RM' in method_names:
            print(method_name)
            if method_name == 'Reward Machines':
                steps = rm_steps
            else:
                ave_reward = np.concatenate((ave_reward, [ave_reward[-1]]))
                min_reward = np.concatenate((min_reward, [min_reward[-1]]))
                max_reward = np.concatenate((max_reward, [max_reward[-1]]))
                steps = np.concatenate((h_steps, [rm_steps[-1]]))
        else:
            steps = h_steps
        plt.plot(steps, ave_reward, color=method_color, label=method_name)    
        plt.fill_between(steps, min_reward, max_reward, color=method_color, alpha=0.2)

    # plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    # plt.title('Reward Averaged over Tasks')
    plt.xlabel('Number of training steps', fontsize=21)
    plt.ylabel('Average reward', fontsize=21)
    if 'RM' in method_names:
        plt.xlim((0, rm_steps[-1]))
        plt.ylim(-250, 30)
    else:
        plt.ylim(-210, 30)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.gca().xaxis.offsetText.set_fontsize(17)


    directory = Path(__file__).parent / 'dataset' / 'lunchbox'
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = 'results_averaged_over_tasks.png'
    path_name = directory / file_name
    legend_path = directory / 'legend.png'

    plt.savefig(path_name, bbox_inches='tight')

    legend = plt.legend(ncol=5, bbox_to_anchor=(0.5, -0.2), framealpha=1, frameon=True)

    export_legend(legend, filename=legend_path)

def plot_data_per_task():
    for i, task_name in enumerate(task_names):
        fig = plt.figure()
        method_ave_rewards, method_min_rewards, \
        method_max_rewards = get_plot_data_for_task(i, task_name)

        for i, (method_name, method_color, ave_reward, min_reward, max_reward) in enumerate(zip(
                method_plot_names, method_colors, method_ave_rewards, method_min_rewards, method_max_rewards)):
            if 'RM' in method_names:
                if method_name == 'Reward Machines':
                    steps = rm_steps
                else:
                    ave_reward = np.concatenate((ave_reward, [ave_reward[-1]]))
                    min_reward = np.concatenate((min_reward, [min_reward[-1]]))
                    max_reward = np.concatenate((max_reward, [max_reward[-1]]))
                    steps = np.concatenate((h_steps, [rm_steps[-1]]))
            else:
                steps = h_steps
            plt.plot(steps, ave_reward, color=method_color, label=method_name)    
            plt.fill_between(steps, min_reward, max_reward, color=method_color, alpha=0.2)

        # plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))
        plt.tight_layout()
        # plt.title("Reward for {} task".format(task_name))
        plt.xlabel('Number of training steps', fontsize=21)
        plt.ylabel('Average reward', fontsize=21)
        if 'RM' in method_names:
            plt.xlim((0, rm_steps[-1]))
            plt.ylim(-250, 30)
        else:
            plt.ylim(-210, 30)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.gca().xaxis.offsetText.set_fontsize(17)


        directory = Path(__file__).parent / 'dataset' / 'lunchbox'
        # if directory doesn't exist, create it
        Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = 'results_{}.png'.format(task_name)
        path_name = directory / file_name

        plt.savefig(path_name, bbox_inches='tight')

def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

if __name__ == '__main__':
    plot_data_over_tasks()
    plot_data_per_task()
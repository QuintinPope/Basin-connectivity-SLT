import torch
from slt_measures_helper import Experiment
import time
import pandas as pd
import matplotlib.pyplot as plt

def rlct_sensitivity(model: Experiment, file_name, 
                     verbose=True,
                     noise_min = 4, 
                     noise_max=6.3, 
                     noise_step=0.3, 
                     iter_min = -25, 
                     iter_max = -6, 
                     iter_step = 1):
    '''
    Suggested use: Use this function and the next to find straight & vertical 
    noise standard deviation lines. If you can't find any straight & vertical lines, 
    find the two noise standard deviations where your lines transition from curvy & uncertain
    to straight (not vertical) & certain lines. Then re-run on noise standard deviations between
    those two phases. You'll want to use noise standard deviations which produce straight
    lines as your RLCT estimate.
    '''

    sensitivity_data = {
        'SGLD iterations': list(), 
        'SGLD noise std': list(), 
        'Mean RLCT': list(),
        'Lower RLCT CI': list(),
        'Upper RLCT CI': list(),
        'Energy': list(), 
        'Free Energy': list()
    }
    t0 = time.time()
    for sgld_noise_std in torch.arange(noise_min, noise_max, noise_step):
        for sgld_num_iter in torch.arange(iter_min, iter_max, iter_step):
            if verbose:
                print(f"time: {time.time() - t0}")
                print(f"sgld num iter: {sgld_num_iter.item()}")
                print(f"sgld noise std: {sgld_noise_std.item()}")
            model.sgld_num_chains = 10
            model.sgld_num_iter = sgld_num_iter.item()
            model.sgld_noise_std = 10**-sgld_noise_std.item()

            fenergy, energy, rlct, lower, upper = model.compute_fenergy_energy_rlct()

            sensitivity_data['SGLD iterations'].append(sgld_num_iter.item())
            sensitivity_data['SGLD noise std'].append(10**-sgld_noise_std.item())
            sensitivity_data['Mean RLCT'].append(rlct)
            sensitivity_data['Lower RLCT CI'].append(lower)
            sensitivity_data['Upper RLCT CI'].append(upper)
            sensitivity_data['Energy'].append(energy)
            sensitivity_data['Free Energy'].append(fenergy)

    sensitivity_dataframe = pd.DataFrame(sensitivity_data)
    sensitivity_dataframe.to_csv(file_name, index=False)

def sensitivity_plot(datafile: str, max_rlct=10**9, save_plot=False):
    sensitivity_data = pd.read_csv(datafile)
    sensitivity_data = sensitivity_data[sensitivity_data["Mean RLCT"] <= max_rlct]

    fig, ax = plt.subplots()

    for name, group in sensitivity_data.groupby("SGLD noise std"):
        group = group.sort_values(by="SGLd iterations")
        ax.errorbar(group["Mean RLCT"], group["SGLD iterations"], xerr=[group["Mean RLCT"]-group["CI_low"], group["CI_high"]-group["Mean RLCT"]],
                    fmt='o', label='Noise std: {:.1f}'.format(name))  # Format to one decimal place
        
    ax.set_xlabel("Mean RLCT")
    ax.set_ylabel("SGLD Iterations")
    ax.set_xscale('log')

    plt.title("RLCT sensitivity plot")
    plt.legend(bbox_to_anchor=(1.05, 1), log='upper left')
    if save_plot: plt.savefig(datafile[:-4] + ".png", bbox_inches='tight')
    plt.show()
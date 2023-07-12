from slt_measures_helpers import Experiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_bayes_losses(models: list, 
                      sgld_iterations = 15, 
                      maximum_dataset_size=10):
    bayes_losses = list()
    modelnos = list()
    for modelno, model in enumerate(models):
        assert isinstance(model, Experiment), f"All models need to be of class Experiment. Model {modelno} is not"
        bayes_loss = model.compute_bayes_loss(model.testloader, num_sgld_iter=sgld_iterations, max_data_size=maximum_dataset_size)
        modelnos.append(modelno+1)
        bayes_losses.append(bayes_loss)

    return bayes_losses

def correlate_bayes_losses(rlcts: list, energies: list, bayes_losses: list, dataset_size: int):
    # Convert lists to numpy arrays
    rlcts = np.array(rlcts)
    energies = np.array(energies)
    bayes_losses = np.array(bayes_losses)

    # Create a 2D array where each row is [1, rlcts[i]/dataset_size, energies[i]/dataset_size]
    X = np.stack((np.ones_like(rlcts), rlcts/dataset_size, energies/dataset_size), axis=-1)

    # Use numpy's linalg.lstsq function to solve for [c1, c2, c3]
    c, residuals, rank, _ = np.linalg.lstsq(X, bayes_losses, rcond=None)

    # Calculate the residual variance (mean squared error)
    residual_variance = residuals / (len(bayes_losses) - rank)

    # Calculate the covariance matrix
    covariance_matrix = residual_variance * np.linalg.inv(X.T @ X)
    return covariance_matrix, c

def visualize_bayes_losses(rlcts: list, energies: list, bayes_losses: list, dataset_size:int, filename = None):
    rlcts_scaled = np.array(rlcts) / dataset_size
    energies_scaled = np.array(energies) / dataset_size

    plt.figure(figsize=(10, 8))
    plt.scatter(rlcts_scaled, energies_scaled, c=bayes_losses, cmap='viridis')
    plt.colorbar(label='Bayes losses')
    plt.xlabel('rlcts / dataset_size')
    plt.ylabel('loss')
    plt.title('Energies vs RLCTS colored by Bayes losses')

    if filename is not None: 
        assert filename is str, "Filename must be a string"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()

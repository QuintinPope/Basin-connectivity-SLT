import torch
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification, BertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

from connectivity_helpers import *
from train_helpers import *
from get_data import *

parser = argparse.ArgumentParser(description='Model training script for linear basin connectivity experiments.')


parser.add_argument('--dataset', type=str, default='mnli',
                    help='Huggingface dataset name to train the model on. Can also be a path to a saved dataset.')

parser.add_argument('--load_saved_data', type=str, default='true',
                    help='Do we try to load a saved dataset, or re-download the data?')

parser.add_argument('--longest_sequence_allowed', type=int, default=64,
                    help='Ignore datapoints with more than this many tokens.')

parser.add_argument('--test_set_size', type=int, default=10000,
                    help='Number of datapoints to include in the test set.')

parser.add_argument('--use_test_data', type=str, default='true',
                    help='Set to any value but \"true\" in order to switch to using the last test_set_size datapoints from the training data instead.')

parser.add_argument('--make_binary', type=str, default='true',
                    help='Do we convert the datasets to a binary prediction problem? Any value but "true" means we do not.')

parser.add_argument('--model_1_path', type=str,
                    help='Location of first model.')

parser.add_argument('--model_2_path', type=str,
                    help='Location of second model.')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size during training.')

parser.add_argument('--n_interpolation_points', type=int, default=20,
                    help='Number of points to check on the linear path between the models.')

parser.add_argument('--device', type=str, default="cuda:0",
                    help='GPU / CPU device on which experiments will run.')

parser.add_argument('--show_plot', type=str, default='true',
                    help='Do we show a plot of the interpolation losses? Any value but "true" means we do not.')

parser.add_argument('--save_plot_loc', type=str, default='None',
                    help='Where do we save a plot of the interpolation losses? Keep set to \"None\" to disable.')

parser.add_argument('--save_losses_loc', type=str, default='None',
                    help='Where do we save a CSV of the interpolation losses? Keep set to \"None\" to disable.')

parser.add_argument('--print_losses', type=str, default='true',
                    help='Do we print a list of the interpolation losses? Any value but "true" means we do not.')

args = parser.parse_args()


dataset_label = str.lower(args.dataset)
data_path = "data/" + dataset_label
make_binary = str.lower(args.make_binary) == "true"

if dataset_label == "mnli":
    if str(args.load_saved_data) == "true":
        train_data, train_labels, test_data, test_labels = load_data_from_file(data_path, "mnli")
    else:
        train_data, train_labels, test_data, test_labels = get_mnli_dataset(model_name = args.model_name, 
                                                                            longest_sequence_allowed = args.longest_sequence_allowed, 
                                                                            n_test_data = args.test_set_size, 
                                                                            make_binary = make_binary, 
                                                                            data_path = data_path, 
                                                                            device = args.device, 
                                                                            save_data = True)
elif dataset_label == "anli":
    if str(args.load_saved_data) == "true":
        train_data, train_labels, test_data, test_labels = load_data_from_file(data_path, "anli")
    else:
        train_data, train_labels, test_data, test_labels = get_anli_dataset(model_name = args.model_name, 
                                                                            longest_sequence_allowed = args.longest_sequence_allowed, 
                                                                            n_test_data = args.test_set_size, 
                                                                            make_binary = make_binary, 
                                                                            data_path = data_path, 
                                                                            device = args.device, 
                                                                            save_data = True)
if str.lower(args.use_test_data) != "true":
    print("Using training data to evaluate interpolation losses.")
    test_data = train_data[:args.test_set_size]
    test_labels = train_labels[:args.test_set_size]


model_1 = torch.load(args.model_1_path)
model_2 = torch.load(args.model_2_path)

path_losses = check_barrier_heights(model_1, model_2, test_data, test_labels, batch_size = args.batch_size, n_check_points = args.n_interpolation_points, device = args.device)

if str.lower(args.show_plot) == "true" or str.lower(args.save_plot_loc) != "none":
    plt.plot(range(args.n_interpolation_points), path_losses)
    plt.xlabel("Interpolation point number")
    plt.ylabel(("Train" if str.lower(args.use_test_data) != "true" else "Test") + " Loss")
    if str.lower(args.show_plot) == "true":
        plt.show()
    if str.lower(args.save_plot_loc) != "none":
        plt.savefig(args.save_plot_loc)

if str.lower(args.save_losses_loc) != "none":
    (pd.DataFrame(path_losses)).to_csv(args.save_losses_loc)

if str.lower(args.print_losses) == "true":
    print(path_losses)
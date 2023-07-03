import torch
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification, BertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
#import pandas as pd
import argparse
import os

from connectivity_helpers import *
from train_helpers import *
from get_data import *

parser = argparse.ArgumentParser(description='Model training script for linear basin connectivity experiments.')


parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                    help='Huggingface pretrained language model name.')

parser.add_argument('--dataset', type=str, default='mnli',
                    help='Huggingface dataset name to train the model on. Can also be a path to a saved dataset.')

parser.add_argument('--load_saved_data', type=str, default='true',
                    help='Do we try to load a saved dataset first, or re-download the data?')

parser.add_argument('--existing_models_path', type=str, default='existing_models',
                    help='Directory with existing models to regularize away from.')

parser.add_argument('--optimizer', type=str, default='adamw',
                    help='Optimizer for training.')

parser.add_argument('--lr', type=float, default=2e-5,
                    help='Learning rate for training.')

parser.add_argument('--weight_decay', type=float, default=0.01,
                    help='Weight decay during training.')

parser.add_argument('--num_schedule_cycles', type=int, default=1,
                    help='Number of cycles for learning rate scheduler.')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size during training.')

parser.add_argument('--epochs', type=int, default=3,
                    help='Number of training epochs.')

parser.add_argument('--warmup_frac', type=float, default=0.1,
                    help='Fraction of training to spend on learning rate warmup.')

parser.add_argument('--max_interpolation_model_loss', type=float, default=1.0,
                    help='Maximum value of the basin exploration regularization loss (before weighting).')

parser.add_argument('--basin_exploration_loss_weight', type=float, default=1.0,
                    help='Weight of the basin exploration regularization loss.')

parser.add_argument('--device', type=str, default="cuda:0",
                    help='GPU / CPU device on which experiments will run.')

parser.add_argument('--model_save_name', type=str, default="None",
                    help='Name given to saved models. If set to \"None\", model will not be saved.')

parser.add_argument('--model_save_loc', type=str, default="trained_models_outputs",
                    help='Folder in which models are saved.')


args = parser.parse_args()


model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(args.device)

if str.lower(args.dataset) == "mnli":
    data_path = "data/mnli"
    if str(args.load_saved_data) == "true":
        train_data = torch.load("../" + data_path + "/mnli_training_data.pth")
        train_labels = torch.load("../" + data_path + "/mnli_training_labels.pth")
        test_data = torch.load("../" + data_path + "/mnli_testing_data.pth")
        test_labels = torch.load("../" + data_path + "/mnli_testing_labels.pth")
    else:
        train_data, train_labels, test_data, test_labels = get_mnli_dataset(model_name = "bert-base-uncased", 
                                                                            longest_sequence_allowed = 64, 
                                                                            n_test_data = 10000, 
                                                                            make_binary = True, 
                                                                            data_path = data_path, 
                                                                            device = args.device, 
                                                                            save_data = True)

existing_models = [torch.load("../" + args.existing_models_path + "/" + old_model_name) for old_model_name in os.listdir("../" + args.existing_models_path)]


model, best_model, best_model_test_loss, train_losses, train_accuracies, test_losses, test_accuracies = train_new_model(model,
                                                                                     train_data,
                                                                                     train_labels,
                                                                                     test_data,
                                                                                     test_labels,
                                                                                     existing_models = [],
                                                                                     optimizer = args.optimizer,
                                                                                     lr = args.lr,
                                                                                     weight_decay = args.weight_decay,
                                                                                     num_schedule_cycles = args.num_schedule_cycles,
                                                                                     batch_size = args.batch_size,
                                                                                     epochs = args.epochs,
                                                                                     warmup_frac = args.warmup_frac,
                                                                                     max_interpolation_model_loss = args.max_interpolation_model_loss,
                                                                                     basin_exploration_loss_weight = args.basin_exploration_loss_weight,
                                                                                     device = args.device)
if args.model_save_name != "None":
    torch.save(model, "../" + args.model_save_loc + "/" + args.model_save_name + ".pth")

print(test_accuracies)
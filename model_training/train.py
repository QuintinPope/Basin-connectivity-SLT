import torch
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification, BertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import copy
import matplotlib.pyplot as plt
import torch_optimizer as optim
import argparse
import os

from connectivity_helpers import *
from train_helpers import *



def train_new_model(model,
                    train_data,
                    train_labels,
                    test_data,
                    test_labels,
                    existing_models = [],
                    optimizer = "adamw",
                    lr = 2e-3,
                    weight_decay = 0,
                    batch_size = 128,
                    epochs = 3,
                    warmup_frac = 0.1,
                    num_schedule_cycles = 1,
                    max_interpolation_model_loss = 1,
                    basin_exploration_loss_weight = 1,
                    device = "cuda:0",
                    test_each_update_step = False):
    all_train_losses = []
    all_test_losses = []
    all_train_accuracies = []
    all_test_accuracies = []

    n_train_data = len(train_data)
    steps_per_epoch = n_train_data // batch_size + (n_train_data % batch_size > 0)
    total_steps = steps_per_epoch * epochs

    if str.lower(optimizer) == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
        optimizer_order = 1

    schedule = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                               num_warmup_steps = int(total_steps * warmup_frac),
                                                                               num_training_steps = total_steps,
                                                                               num_cycles = num_schedule_cycles)
    best_model = copy.deepcopy(model)
    best_model_test_loss = 1000000

    output_model = AutoModelForSequenceClassification.from_config(model.config).to(device)

    for epoch in range(epochs):
        epoch_total_train_loss = 0
        epoch_train_n_correct = 0
        for step in range(steps_per_epoch):
            batch_start = step * batch_size
            batch_end = min((step + 1) * batch_size, n_train_data)
            batch_data = train_data[batch_start: batch_end]
            batch_labels = train_labels[batch_start: batch_end]
            outputs = model(batch_data, labels = batch_labels)
            classification_loss = outputs.loss

            basin_exploration_loss = 0
            for prior_model in existing_models:
                prior_model_weight = torch.rand(1).item()
                linear_interpolation_sample_model = linear_model_mix(model, prior_model, prior_model_weight, output_model, device)
                prior_model_outputs = linear_interpolation_sample_model(batch_data, labels = batch_labels)
                interpolation_model_classification_loss = min(prior_model_outputs.loss, max_interpolation_model_loss)
                basin_exploration_loss += interpolation_model_classification_loss
            basin_exploration_loss /= max(1, len(existing_models))

            loss = classification_loss - basin_exploration_loss_weight * basin_exploration_loss
            if optimizer_order == 1:
                loss.backward()
            elif optimizer_order == 2:
                loss.backward(create_graph = True)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            n_correct = torch.sum(predictions == batch_labels).item()
            #return n_correct, logits, predictions, batch_labels
            epoch_total_train_loss += classification_loss.item() * len(batch_labels)
            epoch_train_n_correct += n_correct

            #if test_each_update_step:
            #    step_test_loss, step_test_accuracy = eval_model(model, test_data, test_labels, batch_size = batch_size)
            #    if step_test_loss < best_model_test_loss:
            #        best_model = copy.deepcopy(model)
            #        best_model_test_loss = step_test_loss
            #        print("Updated best model:")
            #        print("New test loss    :", round(step_test_loss, 5))
            #        print("New test accuracy:", round(step_test_accuracy, 5))
        train_loss = epoch_total_train_loss / n_train_data
        train_accuracy = epoch_train_n_correct / n_train_data

        test_loss, test_accuracy = eval_model(model, test_data, test_labels, batch_size = batch_size)
        print("###########################")
        print("Epoch", epoch, ":")
        print("Train loss.   :", round(train_loss, 5))
        print("Train accuracy:", round(train_accuracy, 5))
        print("Test loss.    :", round(test_loss, 5))
        print("Test accuracy :", round(test_accuracy, 5))

        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)
        all_train_accuracies.append(train_accuracy)
        all_test_accuracies.append(test_accuracy)

        if test_loss < best_model_test_loss:
            best_model = copy.deepcopy(model)
            best_model_test_loss = test_loss
    return model, best_model, best_model_test_loss, all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies


parser = argparse.ArgumentParser(description='Model training script for linear basin connectivity experiments.')


# Required positional argument
parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                    help='Huggingface pretrained language model name')

parser.add_argument('--dataset', type=str, default='mnli',
                    help='Huggingface dataset name to train the model on')

parser.add_argument('--existing_models_path', type=str, default='../existing_models',
                    help='Directory with existing models to regularize away from')

parser.add_argument('--optimizer', type=str, default='adamw',
                    help='Optimizer for training')

parser.add_argument('--lr', type=float, default=5e-5,
                    help='Learning rate for training')

parser.add_argument('--weight_decay', type=float, default=0.1,
                    help='Weight decay during training')

parser.add_argument('--num_schedule_cycles', type=int, default=5,
                    help='Number of cycles for learning rate scheduler')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size during training')

parser.add_argument('--epochs', type=int, default=3,
                    help='Number of training epochs')

parser.add_argument('--warmup_frac', type=float, default=0.1,
                    help='Fraction of training to spend on learning rate warmup')

parser.add_argument('--max_interpolation_model_loss', type=float, default=1.0,
                    help='Maximum value of the basin exploration regularization loss (before weighting)')

parser.add_argument('--basin_exploration_loss_weight', type=float, default=1.0,
                    help='Weight of the basin exploration regularization loss')

parser.add_argument('--device', type=str, default="cuda:0",
                    help='GPU / CPU device on which experiments will run')

args = parser.parse_args()


model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

if str.lower(args.dataset) == "mnli":
    train_data, train_labels, test_data, test_labels = get_mnli_dataset(model_name = "bert-base-uncased", 
                                                                        longest_sequence_allowed = 64, 
                                                                        n_test_data = 10000, 
                                                                        make_binary = True, 
                                                                        data_path = "data/mnli", 
                                                                        device = args.device, 
                                                                        save_data = True)
existing_models = [torch.load("../" + args.existing_models_path + "/" + old_model_name) for path in os.listdir("../" + args.existing_models_path)]



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


print(test_accuracies)
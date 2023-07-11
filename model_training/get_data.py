import torch
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd

def get_mnli_dataset(model_name = "bert-base-uncased", 
                     longest_sequence_allowed = 64, 
                     n_test_data = 10000, 
                     make_binary = True, 
                     data_path = "data/mnli", 
                     device = "cuda:0", 
                     save_data = True,
                     data_label = "mnli"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mnli_dataset = load_dataset("multi_nli")
    mnli_pandas = mnli_dataset['train'].to_pandas()[["premise", "hypothesis", "label"]]
    mnli_pandas = mnli_pandas.append(mnli_dataset['validation_matched'].to_pandas()[["premise", "hypothesis", "label"]])
    mnli_pandas = mnli_pandas.append(mnli_dataset['validation_mismatched'].to_pandas()[["premise", "hypothesis", "label"]])
    mnli_pandas = mnli_pandas.sample(frac = 1, random_state=42)

    train_data, train_labels, test_data, test_labels = filter_data(mnli_pandas, 
                                                                   tokenizer,
                                                                   longest_sequence_allowed,
                                                                   make_binary,
                                                                   n_test_data,
                                                                   device)
    if save_data:
        save_data_to_file(train_data, 
                          train_labels, 
                          test_data, 
                          test_labels, 
                          data_path,
                          data_label)
    return train_data, train_labels, test_data, test_labels

def get_anli_dataset(model_name = "bert-base-uncased", 
                     longest_sequence_allowed = 64, 
                     n_test_data = 10000, 
                     make_binary = True, 
                     data_path = "data/anli", 
                     device = "cuda:0", 
                     save_data = True,
                     data_label = "anli"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    anli_dataset = load_dataset("anli")
    anli_pandas = anli_dataset['train_r3'].to_pandas()[["premise", "hypothesis", "label"]]
    anli_pandas = anli_pandas.append(anli_dataset['test_r3'].to_pandas()[["premise", "hypothesis", "label"]])
    anli_pandas = anli_pandas.append(anli_dataset['dev_r3'].to_pandas()[["premise", "hypothesis", "label"]])
    anli_pandas = anli_pandas.sample(frac = 1, random_state=42)

    train_data, train_labels, test_data, test_labels = filter_data(anli_pandas, 
                                                                   tokenizer,
                                                                   longest_sequence_allowed,
                                                                   make_binary,
                                                                   n_test_data,
                                                                   device)
    if save_data:
        save_data_to_file(train_data, 
                          train_labels, 
                          test_data, 
                          test_labels, 
                          data_path,
                          data_label)
    return train_data, train_labels, test_data, test_labels

def save_data_to_file(train_data, 
                      train_labels, 
                      test_data, 
                      test_labels, 
                      path,
                      dataset_label):
    data_save_relative_path = "../" + path + "/" + dataset_label
    print("Attempting to save data to relative path: \"" + data_save_relative_path + "_[stuff].pth" + "\"")
    torch.save(train_data, data_save_relative_path + "_training_data.pth")
    torch.save(train_labels, data_save_relative_path + "_training_labels.pth")
    torch.save(test_data, data_save_relative_path + "_testing_data.pth")
    torch.save(test_labels, data_save_relative_path + "_testing_labels.pth")

def load_data_from_file(path, dataset_label):
    data_load_relative_path = "../" + path + "/" + dataset_label
    print("Attempting to load data from relative path: \"" + data_load_relative_path + "_[stuff].pth" + "\"")
    train_data = torch.load(data_load_relative_path + "_training_data.pth")
    train_labels = torch.load(data_load_relative_path + "_training_labels.pth")
    test_data = torch.load(data_load_relative_path + "_testing_data.pth")
    test_labels = torch.load(data_load_relative_path + "_testing_labels.pth")
    return train_data, train_labels, test_data, test_labels

def tokenize_data(data, tokenizer, max_length = 128):
    text_data = []
    for i in range(len(data)):
        current_text = data.iloc[i]["premise"] + " [SEP] " + data.iloc[i]["hypothesis"]
        text_data.append(current_text)
    return torch.tensor(tokenizer.batch_encode_plus(text_data, padding = "longest", max_length = max_length, truncation = True)['input_ids'])

def filter_data(data, 
                tokenizer,
                longest_sequence_allowed = 64,
                make_binary = True,
                n_test_data = 10000,
                device = "cpu"):
    current_tokens = tokenize_data(data, tokenizer, max_length = longest_sequence_allowed + 10)
    inputs_too_long_flag = torch.sum(current_tokens > 0, dim = 1) <= longest_sequence_allowed
    inputs_short_enough_entries = [i for i in range(len(data)) if inputs_too_long_flag[i]]

    short_mnli_data = data.iloc[inputs_short_enough_entries]
    short_mnli_labels = torch.from_numpy(data.iloc[inputs_short_enough_entries]["label"].values)
    if make_binary:
        binary_labels_selector_flag = short_mnli_labels != 1
        inputs_only_entailment_or_contradiction_entries = [i for i in range(len(short_mnli_labels)) if binary_labels_selector_flag[i]]
        selected_labels = (short_mnli_labels[inputs_only_entailment_or_contradiction_entries] / 2).to(int)
        selected_tokens = tokenize_data(short_mnli_data.iloc[inputs_only_entailment_or_contradiction_entries], tokenizer)
    else:
        selected_labels = short_mnli_labels
        selected_tokens = tokenize_data(short_mnli_data, tokenizer)

    train_set_size = len(selected_tokens) - n_test_data
    random_indices_perm = torch.randperm(len(selected_tokens))
    train_set_selection_indices = random_indices_perm[:train_set_size]
    test_set_selection_indices = random_indices_perm[train_set_size:]

    train_data = selected_tokens[train_set_selection_indices].to(device)
    train_labels = selected_labels[train_set_selection_indices].to(device)
    test_data = selected_tokens[test_set_selection_indices].to(device)
    test_labels = selected_labels[test_set_selection_indices].to(device)
    return train_data, train_labels, test_data, test_labels
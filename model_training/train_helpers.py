import torch
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification
import copy

def eval_model(model, data, labels, batch_size = 128):
    n_data = len(data)
    steps = n_data // batch_size + (n_data % batch_size > 0)
    total_loss = 0
    total_n_correct = 0
    with torch.no_grad():
        for step in range(steps):
            batch_start = step * batch_size
            batch_end = min((step + 1) * batch_size, n_data)
            batch_data = data[batch_start: batch_end]
            batch_labels = labels[batch_start: batch_end]
            outputs = model(batch_data, labels = batch_labels)
            loss = outputs.loss
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=1)
            n_correct = torch.sum(predictions == batch_labels).item()
            total_loss += loss.item() * len(batch_labels)
            total_n_correct += n_correct
    return total_loss / n_data, total_n_correct / n_data


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
    best_model = AutoModelForSequenceClassification.from_config(model.config).to(device)
    #best_model = copy.deepcopy(model)
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
            loss = classification_loss
            
            if optimizer_order == 1:
                loss.backward()
            elif optimizer_order == 2:
                loss.backward(create_graph = True)
            
            for prior_model in existing_models:
                prior_model_weight = torch.rand(1).item()
                linear_interpolation_sample_model = linear_model_mix(model, prior_model, prior_model_weight, output_model, device)
                prior_model_outputs = linear_interpolation_sample_model(batch_data, labels = batch_labels)

                if interpolation_model_classification_loss > max_interpolation_model_loss:
                    interpolation_model_classification_loss = interpolation_model_classification_loss * 0 # should be the same as just setting it equal to a constant, in terms of gradients

                interpolation_model_classification_loss.backward(retain_graph = True)
                grads = list()
                for param in linear_interpolation_sample_model.parameters():
                    grads.append(param.grad.clone())

                for param, grad in zip(model.parameters(), grads):
                    param.grad += - basin_exploration_loss_weight * grad * 4*(prior_model_weight)*(1-prior_model_weight) / len(existing_models) # Strange prior model weight polynomial made to not punish model for being good.

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

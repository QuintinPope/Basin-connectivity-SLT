import torch
import transformers

def eval_model(model, data, labels, batch_size = 128):
    model = model.eval()
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
    model = model.train()
    return total_loss / n_data, total_n_correct / n_data

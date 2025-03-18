import torch


def compute_confusion_matrix(targets, predictions, num_classes=10):
    """Helper function to compute confusion matrix"""
    if isinstance(targets, list):
        targets = torch.tensor(targets)
    if isinstance(predictions, list):
        predictions = torch.tensor(predictions)

    # Create a confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # Fill the confusion matrix
    for t, p in zip(targets, predictions):
        conf_matrix[t, p] += 1

    return conf_matrix


def compute_f1_score(targets, predictions, num_classes=10, average="macro"):
    """Compute F1 score"""

    if isinstance(targets, list):
        targets = torch.tensor(targets)
    if isinstance(predictions, list):
        predictions = torch.tensor(predictions)

    # Compute confusion matrix
    conf_matrix = compute_confusion_matrix(targets, predictions, num_classes)

    # Extract true positives, false positives, and false negatives
    true_positives = torch.diag(conf_matrix)
    false_positives = torch.sum(conf_matrix, dim=0) - true_positives
    false_negatives = torch.sum(conf_matrix, dim=1) - true_positives

    # Compute precision and recall
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)

    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Average the F1 scores
    if average == "macro":
        return torch.mean(f1).item()
    else:
        return f1


def evaluate_metrics(model, test_loader, device, num_classes=10):
    """Eval model performance with multiple metrics"""

    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    # Calculate accuracy
    accuracy = 100 * correct / total

    # Calculate macro F1 score
    macro_f1 = 100 * compute_f1_score(
        all_targets, all_preds, num_classes=num_classes, average="macro"
    )

    # Calculate confusion matrix
    conf_matrix = compute_confusion_matrix(
        all_targets, all_preds, num_classes=num_classes
    )

    return {"accuracy": accuracy, "macro_f1": macro_f1, "conf_matrix": conf_matrix}

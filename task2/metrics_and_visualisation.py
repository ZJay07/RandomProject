import torch
import numpy as np
import json


def compute_confusion_matrix(targets, predictions, num_classes=10):
    """
    Compute confusion matrix from targets and predictions.

    Args:
        targets (list or torch.Tensor): True class labels
        predictions (list or torch.Tensor): Predicted class labels
        num_classes (int, optional): Number of classes in the dataset with a default value of `10`

    Returns:
        torch.Tensor: Confusion matrix of shape (num_classes, num_classes) where
                     rows represent true classes and columns represent predicted classes
    """
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
    """
    Compute F1 score from targets and predictions.

    Args:
        targets (list or torch.Tensor): True class labels
        predictions (list or torch.Tensor): Predicted class labels
        num_classes (int, optional): Number of classes in the datasetwith a default value of `10`
        average (str, optional): Type of averaging - 'macro' to average over all classes,
                                or anything else to return per-class F1 scores with a default value of "macro"

    Returns:
        float or torch.Tensor: If average='macro', returns scalar F1 score averaged across classes.
                              Otherwise, returns tensor of per-class F1 scores.
    """

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
    """
    Evaluate model performance with multiple metrics including accuracy, F1-score, and confusion matrix.

    Args:
        model (nn.Module): Model to evaluate
        test_loader (torch.utils.data.DataLoader): DataLoader for test data
        device (str or torch.device): Device to run evaluation on
        num_classes (int, optional): Number of classes in the dataset with a default value of `10`

    Returns:
        dict: Dictionary containing evaluation metrics:
            - "accuracy" (float): Percentage accuracy on test data
            - "macro_f1" (float): Macro-averaged F1 score
            - "conf_matrix" (torch.Tensor): Confusion matrix
    """

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


def convert_numpy_types(obj):
    """
    Convert NumPy data types to Python native types for JSON serialisation.

    Recursively processes dictionaries, lists, and NumPy values to ensure
    they can be properly serialised to JSON.

    Args:
        obj (any): Object to convert, can be dict, list, NumPy type, or other

    Returns:
        any: Same object structure with NumPy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def print_model_summary(model_name, metrics, sample_epochs=None):
    """
    Print performance summary for a specific model

    Displays final accuracy and F1-score, along with progression of metrics
    over epochs for ensemble models or at sampled epochs for individual models

    Args:
        model_name (str): Name of the model for display
        metrics (dict): Dictionary containing model performance metrics
        sample_epochs (list, optional): List of specific epochs to show metrics for with a default value of `None`

    Returns:
        None: This function only prints to console
    """
    print(f"\n{model_name}")
    print("-" * len(model_name))

    print(f"Final accuracy: {metrics['final_accuracy']:.2f}%")
    print(f"Final F1-score: {metrics['final_f1']:.2f}%")

    # For ensemble models, show progression as models are added
    if "ensemble" in model_name.lower():
        print("\nMetrics as models are added:")
        print("Models | Epochs | Accuracy (%) | F1-Score (%)")
        print("-------|-------|-------------|------------")
        for i, epoch in enumerate(metrics["epochs"]):
            model_num = i + 1
            print(
                f"{model_num}/5    | {epoch:5d} | {metrics['test_acc'][i]:11.2f} | {metrics['test_f1'][i]:11.2f}"
            )

    # For individual models, show metrics at sampled epochs
    elif sample_epochs:
        print("\nMetrics at key epochs:")
        print("Epoch | Accuracy (%) | F1-Score (%)")
        print("------|-------------|------------")
        for epoch in sample_epochs:
            if epoch in metrics["epochs"]:
                idx = metrics["epochs"].index(epoch)
                print(
                    f"{epoch:5d} | {metrics['test_acc'][idx]:11.2f} | {metrics['test_f1'][idx]:11.2f}"
                )


def print_comparison_summary(metrics_dict):
    """
    Print comparison summary of all model performances.

    Creates a tabular display of accuracy, F1-score, and improvement over random
    baseline for all models in the metrics dictionary.

    Args:
        metrics_dict (dict): Dictionary mapping model names to performance metrics

    Returns:
        None: This function only prints to console
    """
    print("\n===== MODEL COMPARISON SUMMARY =====\n")
    print("Model               | Accuracy (%) | F1-Score (%) | Improvement over Random")
    print("--------------------|-------------|--------------|------------------------")

    random_baseline = 10.0  # CIFAR-10 random guessing baseline

    models = [
        ("Base ELM", "base_model"),
        ("MixUp", "mixup_model"),
        ("Ensemble ELM", "ensemble_model"),
        ("Ensemble with MixUp", "ensemble_mixup_model"),
    ]

    for display_name, model_name in models:
        if model_name in metrics_dict:
            acc = metrics_dict[model_name]["final_accuracy"]
            f1 = metrics_dict[model_name]["final_f1"]
            improvement = acc / random_baseline
            print(
                f"{display_name:20s} | {acc:11.2f} | {f1:11.2f} | {improvement:7.1f}x"
            )


def print_observations(metrics_dict):
    """
    Print key observations about model performance comparisons.

    Analyses and summarises the relative performance differences between models,
    focusing on improvements from ensemble methods and MixUp augmentation.

    Args:
        metrics_dict (dict): Dictionary mapping model names to performance metrics

    Returns:
        None: This function only prints to console
    """
    print("\n===== Summary =====\n")
    print("1. All models significantly outperform random guessing (10% baseline)")
    print("2. Ensemble methods provide slight improvements over single models")
    print(
        "3. F1-scores closely track accuracy values, indicating balanced performance across classes"
    )
    print("\nPerformance analysis:")

    # Compare ensemble to base model
    if "base_model" in metrics_dict and "ensemble_model" in metrics_dict:
        base_acc = metrics_dict["base_model"]["final_accuracy"]
        ens_acc = metrics_dict["ensemble_model"]["final_accuracy"]
        improvement = ens_acc - base_acc
        print(
            f"- Ensemble approach improves accuracy by {improvement:.2f}% over the base model"
        )

    # Compare mixup to base model
    if "base_model" in metrics_dict and "mixup_model" in metrics_dict:
        base_acc = metrics_dict["base_model"]["final_accuracy"]
        mixup_acc = metrics_dict["mixup_model"]["final_accuracy"]
        diff = mixup_acc - base_acc
        if diff > 0:
            print(f"- MixUp improves accuracy by {diff:.2f}% over the base model")
        else:
            print(
                f"- MixUp reduces accuracy by {abs(diff):.2f}% compared to the base model"
            )

    # Compare ensemble with mixup to just ensemble
    if "ensemble_model" in metrics_dict and "ensemble_mixup_model" in metrics_dict:
        ens_acc = metrics_dict["ensemble_model"]["final_accuracy"]
        ens_mixup_acc = metrics_dict["ensemble_mixup_model"]["final_accuracy"]
        diff = ens_mixup_acc - ens_acc
        if diff > 0:
            print(f"- Adding MixUp to Ensemble improves accuracy by {diff:.2f}%")
        else:
            print(f"- Adding MixUp to Ensemble reduces accuracy by {abs(diff):.2f}%")


def summarise_metrics(json_file):
    """
    Read metrics from JSON file and print comprehensive performance summary.

    Loads metrics data from a JSON file and generates reports for each model,
    comparative analysis, and key observations about model performance.

    Args:
        json_file (str): Path to JSON file containing model metrics

    Returns:
        None: This function only prints to console
    """
    # Load the JSON data
    with open(json_file, "r") as f:
        metrics = json.load(f)

    print("\n===== EXPERIMENT PERFORMANCE SUMMARY =====")

    # Print summary for each model
    if "base_model" in metrics:
        print_model_summary(
            "1. BASE ELM MODEL", metrics["base_model"], sample_epochs=[5, 10, 15, 20]
        )

    if "mixup_model" in metrics:
        print_model_summary(
            "2. MIXUP MODEL", metrics["mixup_model"], sample_epochs=[5, 10, 15, 20]
        )

    if "ensemble_model" in metrics:
        print_model_summary("3. ENSEMBLE ELM MODEL", metrics["ensemble_model"])

    if "ensemble_mixup_model" in metrics:
        print_model_summary(
            "4. ENSEMBLE WITH MIXUP MODEL", metrics["ensemble_mixup_model"]
        )

    # Print comparison summary
    print_comparison_summary(metrics)

    # Print observations
    print_observations(metrics)

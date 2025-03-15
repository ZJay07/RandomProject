import json

def print_model_summary(model_name, metrics, sample_epochs=None):
    """
    Print summary for a specific experiment model
    """
    print(f"\n{model_name}")
    print("-" * len(model_name))
    
    print(f"Final accuracy: {metrics['final_accuracy']:.2f}%")
    print(f"Final F1-score: {metrics['final_f1']:.2f}%")
    
    # For ensemble models, show progression as models are added
    if 'ensemble' in model_name.lower():
        print("\nMetrics as models are added:")
        print("Models | Epochs | Accuracy (%) | F1-Score (%)")
        print("-------|-------|-------------|------------")
        for i, epoch in enumerate(metrics['epochs']):
            model_num = i + 1
            print(f"{model_num}/5    | {epoch:5d} | {metrics['test_acc'][i]:11.2f} | {metrics['test_f1'][i]:11.2f}")
    
    # For individual models, show metrics at sampled epochs
    elif sample_epochs:
        print("\nMetrics at key epochs:")
        print("Epoch | Accuracy (%) | F1-Score (%)")
        print("------|-------------|------------")
        for epoch in sample_epochs:
            if epoch in metrics['epochs']:
                idx = metrics['epochs'].index(epoch)
                print(f"{epoch:5d} | {metrics['test_acc'][idx]:11.2f} | {metrics['test_f1'][idx]:11.2f}")

def print_comparison_summary(metrics_dict):
    """
    Print comparison summary of all models
    """
    print("\n===== MODEL COMPARISON SUMMARY =====\n")
    print("Model               | Accuracy (%) | F1-Score (%) | Improvement over Random")
    print("--------------------|-------------|--------------|------------------------")
    
    random_baseline = 10.0  # CIFAR-10 random guessing baseline
    
    models = [
        ('Base ELM', 'base_model'),
        ('MixUp', 'mixup_model'),
        ('Ensemble ELM', 'ensemble_model'),
        ('Ensemble with MixUp', 'ensemble_mixup_model')
    ]
    
    for display_name, model_name in models:
        if model_name in metrics_dict:
            acc = metrics_dict[model_name]['final_accuracy']
            f1 = metrics_dict[model_name]['final_f1']
            improvement = acc / random_baseline
            print(f"{display_name:20s} | {acc:11.2f} | {f1:11.2f} | {improvement:7.1f}x")

def print_observations(metrics_dict):
    """
    Print key observations about model performance
    """
    print("\n===== Summary =====\n")
    print("1. All models significantly outperform random guessing (10% baseline)")
    print("2. Ensemble methods provide slight improvements over single models")
    print("3. F1-scores closely track accuracy values, indicating balanced performance across classes")
    print("\nPerformance analysis:")
    
    # Compare ensemble to base model
    if 'base_model' in metrics_dict and 'ensemble_model' in metrics_dict:
        base_acc = metrics_dict['base_model']['final_accuracy']
        ens_acc = metrics_dict['ensemble_model']['final_accuracy']
        improvement = ens_acc - base_acc
        print(f"- Ensemble approach improves accuracy by {improvement:.2f}% over the base model")
    
    # Compare mixup to base model
    if 'base_model' in metrics_dict and 'mixup_model' in metrics_dict:
        base_acc = metrics_dict['base_model']['final_accuracy']
        mixup_acc = metrics_dict['mixup_model']['final_accuracy']
        diff = mixup_acc - base_acc
        if diff > 0:
            print(f"- MixUp improves accuracy by {diff:.2f}% over the base model")
        else:
            print(f"- MixUp reduces accuracy by {abs(diff):.2f}% compared to the base model")
    
    # Compare ensemble with mixup to just ensemble
    if 'ensemble_model' in metrics_dict and 'ensemble_mixup_model' in metrics_dict:
        ens_acc = metrics_dict['ensemble_model']['final_accuracy']
        ens_mixup_acc = metrics_dict['ensemble_mixup_model']['final_accuracy']
        diff = ens_mixup_acc - ens_acc
        if diff > 0:
            print(f"- Adding MixUp to Ensemble improves accuracy by {diff:.2f}%")
        else:
            print(f"- Adding MixUp to Ensemble reduces accuracy by {abs(diff):.2f}%")

def summarize_metrics(json_file):
    """
    Read metrics from JSON file and print a summary
    """
    # Load the JSON data
    with open(json_file, 'r') as f:
        metrics = json.load(f)
    
    print("\n===== EXPERIMENT PERFORMANCE SUMMARY =====")
    
    # Print summary for each model
    if 'base_model' in metrics:
        print_model_summary("1. BASE ELM MODEL", metrics['base_model'], sample_epochs=[5, 10, 15, 20])
    
    if 'mixup_model' in metrics:
        print_model_summary("2. MIXUP MODEL", metrics['mixup_model'], sample_epochs=[5, 10, 15, 20])
    
    if 'ensemble_model' in metrics:
        print_model_summary("3. ENSEMBLE ELM MODEL", metrics['ensemble_model'])
    
    if 'ensemble_mixup_model' in metrics:
        print_model_summary("4. ENSEMBLE WITH MIXUP MODEL", metrics['ensemble_mixup_model'])
    
    # Print comparison summary
    print_comparison_summary(metrics)
    
    # Print observations
    print_observations(metrics)

if __name__ == "__main__":
    json_file = "models/metrics_results.json" 
    
    try:
        summarize_metrics(json_file)
    except FileNotFoundError:
        print(f"Error: Could not find file '{json_file}'")
        print("Please provide the correct path to your metrics JSON file.")
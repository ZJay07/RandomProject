### Task 1
Everything required in task1 is in the task1 folder, print statemnts at the
```bash
if __name__ == "__main__":
```
code blocks

### Task 2
Some experiments may take awhile (usually due to the large search space), but generated metrics are in the `task2/metrics` directory for hyperparameter search, task2a's elm comparisons, task2a's `fit_elm_ls vs fit_elm_sgd`, task2a's random search for `fit_elm_ls` and `task2/logs` directory for summary of regularisation results 

1. mixup.png is already generated in task2 directory, will be able to regenerate with the command
```bash
python task2/mix_up.py
```
2. Everything in task2 is in the task.py file all experiments can be run with
```bash
python task2/task.py
```

3. `results.png is` saved in the `montage_results` directory, can be reproduced within by running the command above for `task.py`

4. Running `python task2/task2a.py` will run task2a specific experiments, but pre ran metrics are already within the metric or logs directory. Train model weights are in the `task2/models` directory

5. `new_results.png` is in the `montage_results` directory as well, but can be reproduced in `task2a.py`
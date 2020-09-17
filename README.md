# Automl for chexpert

## Run
Execute with `python auto_chexpert.py`. In the very first run please also specify `--prepare_data True`, subsequent runs should have `--prepare_data False` (the default).

## Prepare data
Dataset is prepared by calling the `prepare_dataset()` function from `auto_chexpert.py`

## TODO
* Extend to multilabel
* Replace Ray Tune with Optuna

## Current limitations 
* categorical labels
* validation is not done on the oficial validation set but on a portion of the official training set, due to lack of categorical labels in the official validation selection

## Outputs
Example output:

```[[0, 0, 2], 600.0, {"submitted": 1595509964.8870783, "started": 1595509964.8872344, "finished": 1595510555.3876612}, {"loss": -0.9395823369998406, "info": {"loss": 0.05765587463974953, "model_parameters": 1812103.0, "train_balanced_accuracy": 0.9388869320183163, "train_pac_metric": -2.4625731609448267, "lr_scheduler_converged": 0.0,"```**val_balanced_accuracy**```": ```**0.9395823369998406**```, "val_pac_metric": -2.4530017490351814}}, null]```

with the design:

```[[0, 0, 2], {"CreateDataLoader:batch_size": 180, "Imputation:strategy": "mean", "InitializationSelector:initialization_method": "default", "InitializationSelector:initializer:initialize_bias": "No", "LearningrateSchedulerSelector:lr_scheduler": "plateau", "LossModuleSelector:loss_module": "cross_entropy_weighted", "NetworkSelector:network": "resnet", "NormalizationStrategySelector:normalization_strategy": "minmax", "OptimizerSelector:optimizer": "adamw", "PreprocessorSelector:preprocessor": "fast_ica", "ResamplingStrategySelector:over_sampling_method": "smote", "ResamplingStrategySelector:target_size_strategy": "none", "ResamplingStrategySelector:under_sampling_method": "random", "TrainNode:batch_loss_computation_technique": "mixup", "LearningrateSchedulerSelector:plateau:factor": 0.1517151462726338, "LearningrateSchedulerSelector:plateau:patience": 4, "NetworkSelector:resnet:activation": "relu", "NetworkSelector:resnet:blocks_per_group": 3, "NetworkSelector:resnet:num_groups": 6, "NetworkSelector:resnet:num_units_0": 40, "NetworkSelector:resnet:num_units_1": 60, "NetworkSelector:resnet:use_dropout": false, "NetworkSelector:resnet:use_shake_drop": false, "NetworkSelector:resnet:use_shake_shake": true, "OptimizerSelector:adamw:learning_rate": 0.015022508042355017, "OptimizerSelector:adamw:weight_decay": 0.06738780394335982, "PreprocessorSelector:fast_ica:algorithm": "parallel", "PreprocessorSelector:fast_ica:fun": "exp", "PreprocessorSelector:fast_ica:whiten": false, "ResamplingStrategySelector:smote:k_neighbors": 3, "TrainNode:mixup:alpha": 0.043600999588951694, "NetworkSelector:resnet:num_units_2": 24, "NetworkSelector:resnet:num_units_3": 27, "NetworkSelector:resnet:num_units_4": 203, "NetworkSelector:resnet:num_units_5": 40, "NetworkSelector:resnet:num_units_6": 346}, {}]```

and final fit:

```{"optimized_hyperparameter_config": {"CreateDataLoader:batch_size": 180, "Imputation:strategy": "mean", "InitializationSelector:initialization_method": "default", "InitializationSelector:initializer:initialize_bias": "No", "LearningrateSchedulerSelector:lr_scheduler": "plateau", "LossModuleSelector:loss_module": "cross_entropy_weighted", "NetworkSelector:network": "resnet", "NormalizationStrategySelector:normalization_strategy": "minmax", "OptimizerSelector:optimizer": "adamw", "PreprocessorSelector:preprocessor": "fast_ica", "ResamplingStrategySelector:over_sampling_method": "smote", "ResamplingStrategySelector:target_size_strategy": "none", "ResamplingStrategySelector:under_sampling_method": "random", "TrainNode:batch_loss_computation_technique": "mixup", "LearningrateSchedulerSelector:plateau:factor": 0.1517151462726338, "LearningrateSchedulerSelector:plateau:patience": 4, "NetworkSelector:resnet:activation": "relu", "NetworkSelector:resnet:blocks_per_group": 3, "NetworkSelector:resnet:num_groups": 6, "NetworkSelector:resnet:num_units_0": 40, "NetworkSelector:resnet:num_units_1": 60, "NetworkSelector:resnet:use_dropout": false, "NetworkSelector:resnet:use_shake_drop": false, "NetworkSelector:resnet:use_shake_shake": true, "OptimizerSelector:adamw:learning_rate": 0.015022508042355017, "OptimizerSelector:adamw:weight_decay": 0.06738780394335982, "PreprocessorSelector:fast_ica:algorithm": "parallel", "PreprocessorSelector:fast_ica:fun": "exp", "PreprocessorSelector:fast_ica:whiten": false, "ResamplingStrategySelector:smote:k_neighbors": 3, "TrainNode:mixup:alpha": 0.043600999588951694, "NetworkSelector:resnet:num_units_2": 24, "NetworkSelector:resnet:num_units_3": 27, "NetworkSelector:resnet:num_units_4": 203, "NetworkSelector:resnet:num_units_5": 40, "NetworkSelector:resnet:num_units_6": 346}, "budget": 600.0, "loss": -0.9395823369998406, "info": {"loss": 0.05765587463974953, "model_parameters": 1812103.0, "train_balanced_accuracy": 0.9388869320183163, "train_pac_metric": -2.4625731609448267, "lr_scheduler_converged": 0.0, "```**val_balanced_accuracy**```": ```**0.9395823369998406**```, "val_pac_metric": -2.4530017490351814}}```

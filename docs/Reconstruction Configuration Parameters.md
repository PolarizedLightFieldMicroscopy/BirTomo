# Reconstruction Configuration Key Descriptions

This document describes the various keys and their potential values in the JSON reconstruction configuration file.

---

## General Settings

| Key                         | Description                                                                 | Possible Values           | Default        |
|-----------------------------|-----------------------------------------------------------------------------|---------------------------|----------------|
| `general.num_iterations`     | Number of iterations for the training or reconstruction process.            | Integer (e.g., 200)        | 200            |
| `general.save_freq`          | Frequency for saving intermediate results (in iterations).                  | Integer (e.g., 100)        | 100            |
| `general.output_directory_postfix` | A string appended to the output directory for easier identification.    | String                    | `""` (empty)   |
| `general.notes`              | Additional notes for the configuration run.                                  | String                    | `""` (empty)   |

---

## Learning Rates

| Key                         | Description                                                                 | Possible Values           | Default        |
|-----------------------------|-----------------------------------------------------------------------------|---------------------------|----------------|
| `learning_rates.birefringence` | Learning rate for birefringence optimization.                              | Float (e.g., 1e-4)        | 1e-4           |
| `learning_rates.optic_axis`  | Learning rate for optic axis optimization.                                   | Float (e.g., 1e-1)        | 1e-1           |

---

## Regularization

| Key                         | Description                                                                 | Possible Values           | Default        |
|-----------------------------|-----------------------------------------------------------------------------|---------------------------|----------------|
| `regularization.weight`      | Weight of the regularization term in the loss function.                      | Float (e.g., 0.5)         | 0.5            |
| `regularization.functions`   | List of regularization functions and their associated weights.               | List (e.g., `["function_name", weight]`) | N/A  |

---

## File Paths

| Key                         | Description                                                                 | Possible Values           | Default        |
|-----------------------------|-----------------------------------------------------------------------------|---------------------------|----------------|
| `file_paths.initial_volume`  | Filepath for the initial volume, if any.                                     | Filepath (string)          | `null`         |
| `file_paths.saved_rays`      | Filepath for saved rays, if any.                                             | Filepath (string)          | `null`         |
| `file_paths.vox_indices_by_mla_idx` | Filepath for voxel indices mapped by MLA index.                         | Filepath (string)          | `null`         |
| `file_paths.ret_image`       | Filepath for the measured retardance image.                                  | Filepath (string)          | `null`         |
| `file_paths.azim_image`      | Filepath for the measured azimuth image.                                     | Filepath (string)          | `null`         |
| `file_paths.radiometry`      | Filepath for the radiometry data.                                            | Filepath (string)          | `null`         |

---

## Schedulers

| Key                         | Description                                                                 | Possible Values           | Default        |
|-----------------------------|-----------------------------------------------------------------------------|---------------------------|----------------|
| `schedulers.birefringence.type` | Scheduler type for birefringence learning rate.                           | String (e.g., "ReduceLROnPlateau") | "ReduceLROnPlateau"          |
| `schedulers.birefringence.params.mode` | Mode for scheduler, controls if the scheduler is reducing learning rates based on min/max of the loss. | `"min"`, `"max"` | `"min"`    |
| `schedulers.birefringence.params.factor` | Factor by which the learning rate is reduced.                            | Float (e.g., 0.8)         | 0.8            |
| `schedulers.birefringence.params.patience` | Number of epochs with no improvement before reducing learning rate.       | Integer (e.g., 5)         | 5              |
| `schedulers.birefringence.params.threshold` | Threshold for measuring new optimal value.                               | Float (e.g., 1e-6)        | 1e-6           |
| `schedulers.birefringence.params.min_lr` | Minimum learning rate after reductions.                                   | Float (e.g., 1e-8)        | 1e-8           |
| `schedulers.optic_axis.type` | Scheduler type for optic axis learning rate.                                 | String (e.g., "CosineAnnealingWarmRestarts") | "ReduceLROnPlateau" |
| `schedulers.optic_axis.params.T_0` | Number of iterations for the first restart cycle.                            | Integer (e.g., 20)        | N/A             |
| `schedulers.optic_axis.params.T_mult` | Multiplication factor to increase the length of each cycle.                 | Integer (e.g., 2)         | N/A              |
| `schedulers.optic_axis.params.eta_min` | Minimum learning rate during annealing.                                     | Float (e.g., 1e-4)        | N/A           |

---

## NeRF Settings

| Key                         | Description                                                                 | Possible Values           | Default        |
|-----------------------------|-----------------------------------------------------------------------------|---------------------------|----------------|
| `nerf.enabled`               | Boolean flag to enable or disable NeRF mode.                                 | `true`, `false`           | `false`        |
| `nerf.learning_rates.fc1`    | Learning rate for the first NeRF fully-connected layer (fc1).                | Float (e.g., 1e-2)        | 1e-2           |
| `nerf.learning_rates.fc2`    | Learning rate for the second NeRF fully-connected layer (fc2).               | Float (e.g., 1e-4)        | 1e-4           |
| `nerf.learning_rates.fc3`    | Learning rate for the third NeRF fully-connected layer (fc3).                | Float (e.g., 1e-4)        | 1e-4           |
| `nerf.learning_rates.output` | Learning rate for the NeRF output layer.                                     | Float (e.g., 1e-4)        | 1e-4           |
| `nerf.optimizer.type`        | Type of optimizer used in NeRF mode.                                         | String (e.g., "NAdam")    | `"NAdam"`      |
| `nerf.optimizer.betas`       | Betas for momentum terms in the NeRF optimizer.                              | List (e.g., `[0.9, 0.999]`) | `[0.9, 0.999]`|
| `nerf.optimizer.eps`         | Epsilon value for numerical stability in NeRF optimizer.                     | Float (e.g., 1e-7)        | 1e-7           |
| `nerf.optimizer.weight_decay` | Weight decay (L2 regularization) for the NeRF optimizer.                     | Float (e.g., 1e-4)        | 1e-4           |
| `nerf.scheduler.type`        | Scheduler type for NeRF learning rates.                                      | String (e.g., "CosineAnnealingLR") | N/A  |
| `nerf.scheduler.params`      | Parameters for the NeRF scheduler.                                           | Dictionary                   | N/A            |
| `nerf.MLP.hidden_layers`    | Hidden layers for the NeRF MLP.                                               | List (e.g., `[256, 256, 256]`)    | `[256, 256, 256]` |
| `nerf.MLP.num_frequencies`   | Number of frequencies for the NeRF MLP.                                        | Integer (e.g., 10)          | 10             |
| `nerf.MLP.final_layer_bias_birefringence` | Bias for the final layer of the NeRF MLP for birefringence.                | Float (e.g., -0.05)        | -0.05         |
| `nerf.MLP.final_layer_weight_range` | Weight range for the final layer of the NeRF MLP for birefringence.          | List (e.g., `[-0.01, 0.01]`) | `[-0.01, 0.01]` |

---

## Visualization

| Key                         | Description                                                                 | Possible Values           | Default        |
|-----------------------------|-----------------------------------------------------------------------------|---------------------------|----------------|
| `visualization.plot_live`  | Boolean flag to determine whether to plot the reconstruction live.           | `true`, `false`           | `true`         |

---

## Learnables

| Key                         | Description                                                                 | Possible Values           | Default        |
|-----------------------------|-----------------------------------------------------------------------------|---------------------------|----------------|
| `learnables.all_prop_elements` | Boolean flag to indicate if all properties are learned.                     | `true`, `false`           | `false`        |
| `learnables.two_optic_axis_components` | Boolean flag to indicate if two components are used for optic axis.        | `true`, `false`           | `true`         |

---

## Miscellaneous Settings

| Key                         | Description                                                                 | Possible Values           | Default        |
|-----------------------------|-----------------------------------------------------------------------------|---------------------------|----------------|
| `misc.from_simulation`       | Boolean flag to indicate if data comes from simulation or real-world measurements. | `true`, `false`         | `false`        |
| `misc.save_ray_geometry`     | Boolean flag to determine whether to save the ray geometry.                  | `true`, `false`           | `false`        |
| `misc.optimizer`             | Type of optimizer used for training or reconstruction.                       | String (e.g., "Nadam")    | `"Nadam"`      |
| `misc.datafidelity`          | Term used in the loss function for data fidelity.                            | String (e.g., "euler")    | `"euler"`      |
| `misc.mla_rays_at_once`      | Boolean flag to process MLA rays in batches.                                 | `true`, `false`           | `true`         |
| `misc.free_memory_by_del_large_arrays` | Boolean flag to free memory by deleting large arrays when possible.    | `true`, `false`           | `false`        |
| `misc.save_to_logfile`       | Boolean flag to determine whether to save the output to a logfile.            | `true`, `false`           | `true`         |

---

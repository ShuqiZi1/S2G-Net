
# Hyperparameter Sensitivity Analysis Report
## Mamba-GPS ICU Length-of-Stay Prediction

### Experimental Setup
- Total trials: 100
- Best validation R²: 0.3849
- Search space: 12 hyperparameters

### Key Findings

#### Best Hyperparameters:
- mamba_d_model: 128
- mamba_layers: 4
- mamba_d_state: 64
- mamba_dropout: 0.1
- mamba_pooling: last
- gps_layers: 3
- gps_dropout: 0.1
- lg_alpha: 0.7
- lr: 0.0003970703912602035
- batch_size: 64
- clip_grad: 5.0
- sampling_config: [15, 10]

#### Parameter Sensitivity Rankings:
1. gps_dropout: 0.3613
2. mamba_d_state: 0.2752
3. lr: 0.2652
4. clip_grad: 0.2305
5. mamba_dropout: 0.2261
6. mamba_d_model: 0.1865
7. gps_layers: 0.1811
8. mamba_layers: 0.1450
9. lg_alpha: 0.1284
10. batch_size: 0.1098

### Recommendations:
1. Focus on optimizing the top 3-5 most sensitive parameters
2. Parameters with low sensitivity can be fixed to reduce search space
3. Consider parameter interactions for fine-tuning

### Files Generated:
- trial_results.csv: Complete trial results
- best_parameters.json: Best hyperparameters found
- *.png: Visualization plots

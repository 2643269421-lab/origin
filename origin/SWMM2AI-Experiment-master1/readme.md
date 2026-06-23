# SWMM2AI-Experiment
## Project Overview
This project aims to explore the application of deep learning models in urban hydrological simulation. By integrating a mature physical hydrological model (SWMM) with a storm rainfall formula (Chicago hyetograph), a benchmark dataset is constructed. Representative deep learning models (LSTM, GRU, and their variants with attention mechanisms) are systematically trained and evaluated on this dataset. The project also provides a comprehensive performance evaluation on an independent test set, which includes both generated storm events and real rainfall data.

## Background
While traditional hydrological simulation software (such as SWMM) is mechanism-based and clear, it may face computational efficiency bottlenecks in real-time prediction and rapid scenario analysis. In recent years, data-driven deep learning models have provided new approaches for hydrological time series modeling. This project establishes a complete experimental pipeline of "SWMM simulation results -> Deep learning model" to evaluate the potential and limitations of deep learning models in simulating water level time series at specific nodes.

## Main Features
1. Benchmark Dataset Construction
  - Physics-Based Driver: Uses the mature SWMM (Storm Water Management Model) as the foundational physical simulator.
  - Rainfall Input Generation: Employs the Chicago hyetograph storm formula to generate diverse rainfall sequences by adjusting parameters.
  - Target Output Acquisition: Runs the SWMM model to obtain water level (or other hydrological variables) time series at specific network nodes as target outputs.
  - Standardization: Preprocesses and standardizes the generated input-output data to form a dataset suitable for deep learning training.

2. Deep Learning Model Training and Comparison
  - Trains the following representative deep learning models for time series on the unified benchmark dataset:
    - LSTM (Long Short-Term Memory)
    - GRU (Gated Recurrent Unit)
    - LSTM with Attention (incorporates standard attention mechanism)
    - LSTM with Causal Attention (incorporates causal attention mechanism to ensure temporal causality in predictions)

  - Implements a complete training loop, validation, and model saving functionality.

3. Comprehensive Performance Evaluation
  - Independent Test Set:
    - Unseen storm sequences generated using the Chicago hyetograph with different parameters.
    - Real rainfall data obtained from actual rain gauges.

  - Evaluation Metrics: Uses a comprehensive set of metrics to quantify model performance, such as:
    - Root Mean Square Error (RMSE)
    - Mean Absolute Error (MAE)
    - Nash-Sutcliffe Efficiency (NSE)
    - Peak Error, etc.

## Usage
1. Test Generate Rainfall
```python
python swmm/rainfall/test.py
```

2. Test Run SWMM
```python
python swmm/test.py
```

3. Train Model
Modify train.py, edit model_type and model_path.
model_type: SimpleLSTM/SimpleGRU/AttentionLSTM/CausalAttentionLSTM
```python
python train.py
```

4. Prediction
Modify predict.py, edit model_path.
```python
python predict.py
```



## Future Work
1. Introduce more advanced time series models (e.g., Transformer) for comparison
2. Explore the impact of multi-variable inputs (e.g., flow at multiple nodes, antecedent soil moisture) on model performance
3. Investigate the generalization capability of models under extreme rainfall events or different urban underlying surface conditions

## Acknowledgments
- Thanks to the US EPA for developing and maintaining the SWMM model
- Thanks to all contributors of open-source deep learning libraries (e.g., PyTorch)
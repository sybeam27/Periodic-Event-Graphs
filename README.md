# Dynamic Periodic Event Bipartite Graph
paper : Dynamic Periodic Event Bipartite Graphs for Multivariate Time Series Pattern Prediction

## Requirements & Setup
This codebase utilizes Anaconda for managing environmental dependencies. Please follow these steps to set up the environment:

1. **Download Anaconda:** [Click here](https://www.anaconda.com/download) to download Anaconda.

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/peg-repo/periodic-event-graph
   ```

3. **Install Requirements:**
   - Navigate to the cloned repository:
     ```bash
     cd periodic-event-graph
     ```
   - Create a Conda environment from the provided `environment.yaml` file:
     ```bash
     conda env create -f environment.yaml
     ```
   - Activate the Conda environment:
     ```bash
     conda activate periodic-graph
     ```

This will set up the environment required to run the codebase.

## Datasets
Below are the details and download links for datasets used in our experiments.

#### Power Consumption 
The collection of power consumption data for a local community consisting of 50 households and 1 public building. The public building data utilized in the experiment provides consumption profiles for the public building, segmented by appliances. It spans 96 intervals per day at 15-minute intervals, offering a year's worth of data and profiles for 10 appliances. [(link)](https://zenodo.org/records/6778401)  

#### Traffic 
The collection of 48 months (2015-2016) hourly data from the California Department of Transportation. The data describes the road occupancy rates (between 0 and 1) measured by different sensors on San Francisco Bay area freeways. [(link)](https://pems.dot.ca.gov)  

#### Exchange Rate   
The collection of the daily exchange rates of eight foreign countries including Australia, British, Canada, Switzerland, China, Japan, New Zealand and Singapore ranging from 1990 to 2016. [(link)](https://github.com/MTS-BenchMark/MvTS?tab=readme-ov-file)

## Periodic Event Graph Generation
`--period` is for STL algorithm

```python
# power consumption
!python graph_generation.py --dataset_name 'power' --period 4 --motif 5 --cluster 2
# exchange rate
!python graph_generation.py --dataset_name 'exchange' --period 4 --motif 3 --cluster 2
# traffic
!python graph_generation.py --dataset_name 'traffic' --period 4 --motif 3 --cluster 3
```

## Preprocessing
`<dataset_name>` is one of `traffic`, `power`, `exchange`

* periodic event graph **<em>without residual node</em>**
```python
python preprocess_data.py --dataset_name <dataset_name>_peg_wo_residual
```
* periodic event graph **<em>with residual node</em>**
```python
python preprocess_data.py --dataset_name <dataset_name>_peg_w_residual
```
* periodic event graph **<em>with simple residual node</em>**
```python
python preprocess_data.py --dataset_name <dataset_name>_peg_w_simple_residual
```

## Dynamic GNNs Link Prediction
`<dataset_name>` is one of `traffic`, `power`, `exchange`    
and `<dgnn_model>` is one of `JODIE`, `DyRep`, `TGAT`, `TGN`, `GraphMixer`
```python
python train_link_prediction.py --dataset_name <dataset_name>_peg_wo_residual --model_name <dgnn_model> --load_best_configs --num_runs 5 --num_epochs 10
```
#### Optional arguments
```
  --dataset_name                    dataset to be used
  --batch_size                      batch size
  --model_name                      name of the model, note that EdgeBank is only applicable for evaluation
  --gpu GPU                         number of gpu to use
  --num_neighbors                   number of neighbors to sample for each node
  --sample_neighbor_strategy        how to sample historical neighbors
  --time_scaling_factor             the hyperparameter that controls the sampling preference with time interval, a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, it works when sample_neighbor_strategy == time_interval_aware
  --num_walk_heads                  number of heads used for the attention in walk encoder
  --num_heads                       number of heads used in attention layer
  --num_layers                      number of model layers
  --walk_length                     length of each random walk
  --time_gap                        time gap for neighbors to compute node features
  --time_feat_dim                   dimension of the time embedding
  --position_feat_dim               dimension of the position embedding
  --edge_bank_memory_mode           how memory of EdgeBank works
  --time_window_mode                how to select the time window size for time window memory
  --patch_size                      patch size
  --channel_embedding_dim           dimension of each channel embedding
  --max_input_sequence_length       maximal length of the input sequence of each node
  --learning_rate                   learning rate
  --dropout                         dropout rate
  --num_epochs                      number of epochs
  --optimizer                       name of optimizer
  --weight_decay                    weight decay
  --patience                        patience for early stopping
  --val_ratio                       ratio of validation set
  --test_ratio                      ratio of test set
  --num_runs                        number of runs
  --test_interval_epochs            how many epochs to perform testing once
  --negative_sample_strategy        strategy for the negative edge sampling
  --load_best_configs               whether to load the best configurations
```

## Special Thanks to
We extend our gratitude to the authors of the following libraries for generously sharing their source code:

[DyGLib](https://github.com/yule-BUAA/DyGLib/tree/master),
[MASS](https://github.com/tylerwmarrs/mass-ts),
[Peak over Threshold](https://github.com/cbhua/peak-over-threshold)

Your contributions are greatly appreciated.

## Citation
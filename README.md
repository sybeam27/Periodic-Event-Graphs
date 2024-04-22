# Dynamic Periodic Event Bipartite Graph
paper : Dynamic Periodic Event Bipartite Graphs for Multivariate Time Series Pattern Prediction

## Requirements & Setup
This code base utilize anaconda for environmental depecdencies.
So plz download anaconda  [click here](https://www.anaconda.com/download)  

Clone the repository:  
<em>python >= 3.9</em>
```
git clone https://github.com/peg-repo/periodic-event-graph
```

install requirements:
```
cd periodic-event-graph
conda env create -f environment.yaml
conda activate periodic-graph  #virtual environment activate
```

## Datasets
The following are the information and download site link of datasets using experiments.

##### Power Consumption 
The collection of power consumption data for a local community consisting of 50 households and 1 public building. The public building data utilized in the experiment provides consumption profiles for the public building, segmented by appliances. It spans 96 intervals per day at 15-minute intervals, offering a year's worth of data and profiles for 10 appliances. [(link)](https://zenodo.org/records/6778401)  

##### Traffic 
The collection of 48 months (2015-2016) hourly data from the California Department of Transportation. The data describes the road occupancy rates (between 0 and 1) measured by different sensors on San Francisco Bay area freeways. [(link)](https://pems.dot.ca.gov)  

##### Exchange Rate   
The collection of the daily exchange rates of eight foreign countries including Australia, British, Canada, Switzerland, China, Japan, New Zealand and Singapore ranging from 1990 to 2016. [(link)](https://github.com/MTS-BenchMark/MvTS?tab=readme-ov-file)

## Periodic Event Graph Generation

```python
# power consumption
!python graph_generation.py --dataset_name 'power' --period 4 --stride 4 --motif 5 --cluster 2
# exchange rate
!python graph_generation.py --dataset_name 'exchange' --period 4 --stride 4 --motif 3 --cluster 2
# traffic
!python graph_generation.py --dataset_name 'traffic' --period 4 --stride 4 --motif 3 --cluster 3
```

## Preprocessing
`<dataset_name>` is one of `traffic`, `power`, `exchange`
* periodic event graph <em>with residual node</em>
```python
python preprocess_data.py --dataset_name <dataset_name>_peg_w_residual
```
* periodic event graph <em>without residual node</em>
```python
python preprocess_data.py --dataset_name <dataset_name>_peg_wo_residual
```
* periodic event graph <em>with simple residual node</em>
```python
python preprocess_data.py --dataset_name <dataset_name>_peg_w_simple_residual
```

## Dynamic GNNs Link Prediction
`<dataset_name>` is one of `traffic`, `power`, `exchange` and `<dgnn_model>` is one of `JODIE`, `DyRep`, `TGAT`, `TGN`, `GraphMixer`
```python
python train_link_prediction.py --dataset_name <dataset_name>_peg_wo_residual --model_name <dgnn_model> --load_best_configs --num_runs 5 --num_epochs 10
```
#### Optional arguments
```
  --dataset_name DATASET_NAME
                        dataset to be used
  --batch_size BATCH_SIZE
                        batch size
  --model_name {JODIE,DyRep,TGAT,TGN,CAWN,EdgeBank,TCL,GraphMixer,DyGFormer}
                        name of the model, note that EdgeBank is only applicable
                        for evaluation
  --gpu GPU             number of gpu to use
  --num_neighbors NUM_NEIGHBORS
                        number of neighbors to sample for each node
  --sample_neighbor_strategy {uniform,recent,time_interval_aware}
                        how to sample historical neighbors
  --time_scaling_factor TIME_SCALING_FACTOR
                        the hyperparameter that controls the sampling preference
                        with time interval, a large time_scaling_factor tends to
                        sample more on recent links, 0.0 corresponds to uniform
                        sampling, it works when sample_neighbor_strategy ==
                        time_interval_aware
  --num_walk_heads NUM_WALK_HEADS
                        number of heads used for the attention in walk encoder
  --num_heads NUM_HEADS
                        number of heads used in attention layer
  --num_layers NUM_LAYERS
                        number of model layers
  --walk_length WALK_LENGTH
                        length of each random walk
  --time_gap TIME_GAP   time gap for neighbors to compute node features
  --time_feat_dim TIME_FEAT_DIM
                        dimension of the time embedding
  --position_feat_dim POSITION_FEAT_DIM
                        dimension of the position embedding
  --edge_bank_memory_mode {unlimited_memory,time_window_memory,repeat_threshold_memory}
                        how memory of EdgeBank works
  --time_window_mode {fixed_proportion,repeat_interval}
                        how to select the time window size for time window memory
  --patch_size PATCH_SIZE
                        patch size
  --channel_embedding_dim CHANNEL_EMBEDDING_DIM
                        dimension of each channel embedding
  --max_input_sequence_length MAX_INPUT_SEQUENCE_LENGTH
                        maximal length of the input sequence of each node
  --learning_rate LEARNING_RATE
                        learning rate
  --dropout DROPOUT     dropout rate
  --num_epochs NUM_EPOCHS
                        number of epochs
  --optimizer {SGD,Adam,RMSprop}
                        name of optimizer
  --weight_decay WEIGHT_DECAY
                        weight decay
  --patience PATIENCE   patience for early stopping
  --val_ratio VAL_RATIO
                        ratio of validation set
  --test_ratio TEST_RATIO
                        ratio of test set
  --num_runs NUM_RUNS   number of runs
  --test_interval_epochs TEST_INTERVAL_EPOCHS
                        how many epochs to perform testing once
  --negative_sample_strategy {random,historical,inductive}
                        strategy for the negative edge sampling
  --load_best_configs   whether to load the best configurations
```

## Thanks
We are thankful to the authors of
[DyGLib](https://github.com/yule-BUAA/DyGLib/tree/master)
[MASS](https://github.com/tylerwmarrs/mass-ts)
[Peak over Threshold](https://github.com/cbhua/peak-over-threshold)
for making their source codes publicly avaliable.

## Citation
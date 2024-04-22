# Periodic Event Graph
paper : Dynamic Periodic Event Bipartite Graphs for Multivariate Time Series Pattern Prediction

## Requirements

## Datasets
The following are the information and download site link of datasets using experiments.

### Power Consumption 
The collection of power consumption data for a local community consisting of 50 households and 1 public building. The public building data utilized in the experiment provides consumption profiles for the public building, segmented by appliances. It spans 96 intervals per day at 15-minute intervals, offering a year's worth of data and profiles for 10 appliances. [(link)](https://zenodo.org/records/6778401)  

### Traffic 
The collection of 48 months (2015-2016) hourly data from the California Department of Transportation. The data describes the road occupancy rates (between 0 and 1) measured by different sensors on San Francisco Bay area freeways. [(link)](https://pems.dot.ca.gov)  

### Exchange Rate   
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

#### periodic event graph <em>with residual node</em>
```python
# power consumption
python preprocess_data.py --dataset_name power_peg_w_residual
```
#### periodic event graph <em>without residual node</em>
```python
# power consumption
python preprocess_data.py --dataset_name power_peg_wo_residual
```
#### periodic event graph <em>with simple residual node</em>
```python
# power consumption
python preprocess_data.py --dataset_name power_peg_w_simple_residual
```

## Dynamic GNNs Link Prediction
```python
python train_link_prediction.py --dataset_name exchange_peg_wo_residual --model_name DyRep --load_best_configs --num_runs 5 --num_epochs 10
```
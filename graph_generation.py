import os
import sys
import random
import warnings
import argparse

import pandas as pd
import numpy as np
import utils.periodic_event_graph_utils as peg
import utils.event_graph_utils as eg
import utils.simple_periodic_event_graph_utils as speg

warnings.filterwarnings('ignore')
random.seed(1127)

def create_df(dataset_name):
    if dataset_name == 'traffic':
        traffic = pd.read_csv("./data/dataset/traffic.csv")
        traffic.set_index('date', inplace=True)
        df = traffic.iloc[:, 600:620]
        
    elif dataset_name == 'power consumption':
        file_path = './data/dataset/Dataset.xlsx'
        sheet_name = 'PublicBuilding'
        power = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        power = power.drop(['Unnamed: 0', 'Total Consumption', 'AC4', 'Refrigerator'], axis=1)
        power.set_index('Periods', inplace=True)
        power.index = pd.date_range(start='2023-01-01', periods=len(power), freq='15T')
        df = power
        
    elif dataset_name == 'exchange rate':
        exchange = pd.read_csv("./data/dataset/exchange_rate.csv")
        exchange.set_index('date', inplace=True)
        df = exchange
        
    else:
        print("Invalid dataset name. Please choose from 'traffic', 'power consumption', or 'exchange rate'.")
        df = None
    return df

# hyper-parameter
parser = argparse.ArgumentParser(description='Periodic Event Graph Generation')
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset: traffic, power consumption, exchange rate')
parser.add_argument('--period', type=int, default=4, help='STL Period hyper-parameter')
parser.add_argument('--stride', type=int, default=4, help='Window stride hyper-parameter')
parser.add_argument('--motif', type=int, default=3, help='Motifs k hyper-parameter')
parser.add_argument('--cluster', type=int, default=3, help='Minimum cluster size hyper-parameter')
args = parser.parse_args()

df = create_df(args.dataset_name)
print('Periodic Evnet Graph with residual node generating..')
# Periodic Event Graph
# STL
seasonal_data = peg.apply_stl_seasonal(df, args.period)
residual_data = peg.apply_stl_residual(df, args.period)
# period dict
seasonal_dict = peg.dominant_periods(seasonal_data)
size = args.period * int(np.mean(list(seasonal_dict.values())))
# seasonal matrix profile
seasonal_mp_set = peg.matrix_profile_set(seasonal_data, seasonal_dict, args.motif, args.period)
# seasonal event generation
seasonal_event_set = peg.event_generation_set(seasonal_data, seasonal_mp_set, seasonal_dict, args.period, args.cluster)
# residual matrix profile
residual_mp_set = peg.matrix_profile_set(residual_data, seasonal_dict, args.motif, args.period)
# residual event generation
residual_event_set = peg.event_generation_set(residual_data, residual_mp_set, seasonal_dict, args.period, args.cluster)
# seasonal window
seasonal_window = peg.extract_windows_df(df, seasonal_data, seasonal_dict, args.period, args.stride, size)
# residual window
residual_window = peg.extract_windows_df(df, residual_data, seasonal_dict, args.period, args.stride, size)
# pattern node
pattern_node = peg.pattern_matching_df(seasonal_window, seasonal_event_set)
# residual node
residual_node = peg.pattern_matching_df(residual_window, residual_event_set)
# graph generation
peg_graph = peg.event2graph_df(pattern_node, residual_node, seasonal_event_set, residual_event_set)
# save the periodic event graph with residual
peg_graph.to_csv('./data/graph/{}_peg_w_residual.csv'.format(args.dataset_name), sep = ',', index = False)
print('number of periodic event graph with residual node is', len(list(peg_graph['i'].unique())))


print('Periodic Evnet Graph without residual node generating..')
# Periodic Event Graph without residual node
# pattern graph extraction
peg_pattern_graph = peg_graph[::2]
# save the periodic event graph without residual
peg_pattern_graph.to_csv('./data/graph/{}_peg_wo_residual.csv'.format(args.dataset_name), sep = ',', index = False)
print('number of periodic event graph without residual node is', len(list(peg_pattern_graph['i'].unique())))


print('Periodic Evnet Graph with simple residual node generating..')
# Periodic Event Graph with simple residual node
# residual window
residual_window_speg = speg.extract_residual_windows_df(seasonal_window, seasonal_event_set, pattern_node)
# threshold window #extract_threshold_windows_df(df, size, stride, init_level = 0.8):
threshold_window_speg = speg.extract_threshold_windows_df(df, seasonal_data, size, args.stride)
# residual node
residual_node_speg = speg.residual_matching_df(residual_window_speg, threshold_window_speg)
# graph generation
speg_graph = speg.event2graph_df(pattern_node, residual_node_speg, seasonal_event_set)
# save the periodic event graph with simple residual 
speg_graph.to_csv('./data/graph/{}_peg_w_simple_residual.csv'.format(args.dataset_name), sep = ',', index = False)
print('number of periodic event graph with simple residual node is', len(list(speg_graph['i'].unique())))


# # event histogram
# seasonal_map = list(seasonal_event_set.keys())
# residual_map = list(residual_event_set.keys())
# seasonal_mapping = {event: idx for idx, event in enumerate(seasonal_map)}
# residual_mapping = {event: idx for idx, event in enumerate(residual_map)}

# pattern_mapped = pattern_node.copy() 
# for column in pattern_node.columns:
#     pattern_mapped[column] = pattern_node[column].map(seasonal_mapping)

# residual_mapped = residual_node.copy() 
# for column in residual_node.columns:
#     residual_mapped[column] = residual_node[column].map(residual_mapping)
    
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.figure(figsize=(20, 10))

# plt.subplot(2, 1, 1) 
# sns.heatmap(pattern_mapped.T, cmap='tab20b', cbar=False, xticklabels=False, yticklabels=False)
# plt.ylabel('Period Event Node', fontsize=30)

# plt.subplot(2, 1, 2) 
# sns.heatmap(residual_mapped.T, cmap='tab20b_r', cbar=False, xticklabels=False, yticklabels=False)
# plt.ylabel('Period Residual Node', fontsize=30)

# plt.text(0.5, 0.01, '(b) Power Consumption', ha='center', fontsize=60, transform=plt.gcf().transFigure)
# plt.tight_layout()
# plt.figure(dpi=300)
# plt.show()
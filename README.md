
# Learning state-variable relationships in POMCP

## Description

This is the code used to run a single execution of the MRF learning algorithms Sample-based MRF Learning (SL), Maximum-Likelihood Belief-based MRF Learning (MBL) and Weighted-likelihood Belief-based MRF Learning (WBL).
The MRF learning techniques were first presented in "Learning state-variable relationships for improving POMCP performance" by Maddalena Zuccotto, Alberto Castellini and Alessandro Farinelli.
Our POMCP implementation is based on the original BasicPOMCP package implemented in Julia.


## Required interpreters and packages
- Julia 1.5.3
- POMDPs
- RocksampleExtended 
- BasicPOMCP
- POMDPSimulators
- StaticArrays
- DataFrames
- CSV
- DelimitedFiles
- LinearAlgebra
- Random



## Usage

### MRF Learning
To learn the MRF with one of the proposed methods execute the following command in a new terminal from the directory that contains this README.md:

<pre>
julia name_file.jl
</pre>
where `name_file.jl` is
- `SL_learning.jl` to run the Sample-based MRF Learning (SL) algorithm
- `MBL_learning.jl` to run the Maximum-Likelihood Belief-based MRF Learning (MBL) algorithm
- `WBL_learning.jl` to run the Weighted-likelihood Belief-based MRF Learning (WBL) algorithm
- `MBL_introduction.jl` to run the MBL algorithm that uses the learned MRF after a specific number of learning episodes
- `WBL_introduction.jl` to run the WBL algorithm that uses the learned MRF after a specific number of learning episodes
- `MBL_learning_CI.jl` to run the MBL algorithm with the stopping criterion based on confidence intervals


All the runs require a txt file with the true rocks configuration for each episode of the run. For example, in `rocks_config.txt` we have 100 rows of 8 boolean values (8 columns) representing the true state-variable values of each of the 100 episodes in the run. In other words, each rows shows the rocks configuration of an episode.
Notice that rocks configuration in the txt file has to satisfy the distribution defined by the MRF we aim at learning.

Running the proposed learning methods will create a specific folder in the `Test` directory containing
- `mrf_info` folder containing
	- `count_occurences_episode_i.csv` files; each file stores the number of each state-variable (i.e., rock) observed or extracted from the final belief in each episode until the specific episode ("i" is a numeric value that specifies the episode). Not in WBL_learning and WBL_introduction results
	- `values_occurences_episode_i.csv` each file stores the number of times variable 1 had value "x" and variable 2 had value "y" until the specific episode ("i" is a numeric value that specifies the episode). Not in WBL_learning and WBL_introduction results
	- `mrf_episode_i.csv` files; each file stores the MRF learned until the specific episode ("i" is a numeric value that specifies the episode). The csv presents 2*num_rocks rows and 2*num_rocks columns, i.e., 16 rows and 16 columns with the constraint value in each cell.
- `belief_info` folder containing
	- `belief_part_count_ep_i.csv` files; each file stores the count of each particle (i.e., state) in the belief (columns) during each step of the episode (rows)
	- `belief_part_prob_ep_i.csv` files; each file stores the probability of each state in the belief (columns) during each step of the episode (rows)
- `ci_info` folder (only in MBL_learning_CI results) containing
	- `lower_bounds_episode_i.csv` files; each file stores the lower bounds of the specific learning episode ("i" is a numeric value that specifies the episode)
	- `upper_bounds_episode_i.csv` files; each file stores the upper bounds of the specific learning episode ("i" is a numeric value that specifies the episode)
- `mrf_episode_i_used.csv` file; the file stores the MRF introduced. Only in MBL_learning_CI results
- `output_episode_i.csv` files; each file presents the results of the specific episode ("i" is a numeric value that specifies the episode)


In `MBL_introduction.jl` and `WBL_introduction.jl` the episode at which introduce the MRF can be changed setting the value of the following variable:
<pre>
introduce_at = 40		# line 85 in `MBL_introduction.jl` and line 70 in `WBL_introduction.jl`
</pre>

### MRF Usage
To use the MRF during the execution of POMCP execute the following command in a new terminal from the directory that contains this README.md:
<pre>
julia MRF_usage.jl
</pre>

It requires a txt file with the true rocks configuration for each episode of the run and a csv file representing an MRF (learned of given). For example, in `MRF_example.csv` we store an MRF representing contraints among 8 nodes (rocks), considering each possible values of each node. So, considering boolean variables, the csv presents 2*num_nodes rows and 2*num_nodes columns.
Notice that rocks configuration in the txt file has to satisfy the distribution defined by the true MRF.

Running this method will create a specific folder in the `Test` directory containing
- `belief_info` folder containing
	- `belief_part_count_ep_i.csv` files; each file stores the count of each particle (i.e., state) in the belief (columns) during each step of the episode (rows)
	- `belief_part_prob_ep_i.csv` files; each file stores the probability of each state in the belief (columns) during each step of the episode (rows)
- `output_episode_i.csv` files; each file presents the results of the specific episode ("i" is a numeric value that specifies the episode)

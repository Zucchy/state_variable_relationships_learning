module RockSampleExtended

using LinearAlgebra
using POMDPs
using POMDPModelTools
using StaticArrays
using Parameters
using Random
using Compose
using DelimitedFiles
using BasicPOMCP
using POMDPSimulators
using CPUTime
using Random

import BasicPOMCP: POMCPPlanner, POMCPTree, POMCPObsNode, insert_action_node!, insert_obs_node!, estimate_value, action_info, search, simulate

import POMDPSimulators: RolloutSimulator, simulate

import ParticleFilters: UnweightedParticleFilter, ParticleCollection, ParticleFilters, n_particles, particle, update, initialize_belief

export
    RockSamplePOMDP,
    Options,
    RSPos,
    RSState,

    action_info, 
    search, 

    simulate,

    update,

    ParticleSelection,
    ParticleProbability

const RSPos = SVector{2, Int64}

"""
    RSStateExt{K}
Represents the state in a RockSamplePOMDP problem. 
`K` is an integer representing the number of rocks

# Fields
- `pos::RPos` position of the robot
- `rocks::SVector{K, Bool}` the status of the rocks (false=bad, true=good)
- `collected::SVector{K, Bool}`sampled rocks (false=not-sampled, true=sampled)
"""
struct RSState{K}
    pos::RSPos 
    rocks::SVector{K, Bool}
    collected::SVector{K, Bool}
end

@with_kw mutable struct Options
    episode::Int = 1
    episode_step::Int = 1
    log_dir::String = "log"
    knowledge::Int = 1 # "Knowledge level in tree (0=Pure, 1=Legal)"
    parallelisation::Int = 0
    n_cores::Int = 4
    verbose::Int = 0
    mcts_act_sel_ptr::Union{IOStream, Nothing} = nothing
    relations::Int = 1
    n_rand_particles::Int = 100
end

@with_kw mutable struct RockSamplePOMDP{K} <: POMDP{RSState{K}, Int64, Int64}
    map_size::Tuple{Int64, Int64} = (5,5)
    rocks_positions::SVector{K,RSPos} = @SVector([(1,1), (3,3), (4,4)])
    init_pos::RSPos = (1,1)
    sensor_efficiency::Float64 = 10.0
    bad_rock_penalty::Float64 = -10
    good_rock_reward::Float64 = 10.
    exit_reward::Float64 = 10.
    terminal_state::RSState{K} = RSState(RSPos(-1,-1),
                                         SVector{length(rocks_positions),Bool}(falses(length(rocks_positions))),
                                         SVector{length(rocks_positions),Bool}(falses(length(rocks_positions))))
    discount_factor::Float64 = 0.95
    options::Options = Options()
    mrf::Union{Array{Float64,2}, Nothing} = nothing
    rocks_count_per_label_episode = []
    rocks_prob_per_label_episode= []

end

# to handle the case where rocks_positions is not a StaticArray
function RockSamplePOMDP(map_size,
                         rocks_positions,
                         args...
                        )

    k = length(rocks_positions)
    pomdp = RockSamplePOMDP{k}(map_size,
                              SVector{k,RSPos}(rocks_positions),
                              args...)
    if pomdp.options.verbose > 0
        mkpath(pomdp.options.log_dir)
        
        pomdp.options.mcts_act_sel_ptr = open(joinpath(pomdp.options.log_dir, "mcts_action_selection_episode_$(pomdp.options.episode).csv" ), "w")
        
        for a = POMDPs.actions(pomdp)
            write(pomdp.options.mcts_act_sel_ptr, "act_$(a)_n,act_$(a)_v,act_$(a)_ucb")
        end
        
        write(pomdp.options.mcts_act_sel_ptr, "\n")
        
        act_sel_progr_epis = joinpath(pomdp.options.log_dir, "mcts_action_selection_progress_episode_$(pomdp.options.episode)")
        if pomdp.options.verbose > 1 && !isdir(act_sel_progr_epis)
            mkdir(act_sel_progr_epis)
        end

        mcts_desc_epis = joinpath(pomdp.options.log_dir, "mcts_descent_details_episode_$(pomdp.options.episode)")
        if pomdp.options.verbose > 2 && !isdir(mcts_desc_epis)
            mkdir(mcts_desc_epis)
        end

        pf_selection_dir = joinpath(pomdp.options.log_dir, "pf_selection_$(pomdp.options.episode)")
        pf_rejection_dir = joinpath(pomdp.options.log_dir, "pf_rejection_$(pomdp.options.episode)")
        if pomdp.options.verbose == 4 && !isdir(pf_selection_dir) && !isdir(pf_rejection_dir)
            mkdir(pf_selection_dir)
            mkdir(pf_rejection_dir)
        end
    end

    if pomdp.options.relations == 1
        pomdp.mrf = fill(0.25, (2*length(pomdp.rocks_positions),2*length(pomdp.rocks_positions)))
    end
    return pomdp
end

POMDPs.isterminal(pomdp::RockSamplePOMDP, s::RSState) = s.pos == pomdp.terminal_state.pos 
POMDPs.discount(pomdp::RockSamplePOMDP) = pomdp.discount_factor

include("states.jl")
include("actions.jl")
include("transition.jl")
include("observations.jl")
include("reward.jl")
include("visualization.jl")
include("solver.jl")
include("unweighted.jl")
include("rollout.jl")
include("mrf.jl")

end # module

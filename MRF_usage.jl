using POMDPs
using RockSampleExtended
using BasicPOMCP
using POMDPSimulators
using StaticArrays
using DataFrames
using CSV
using DelimitedFiles
using LinearAlgebra
using Random



# Main function to use an MRF (learned offline or given)
function main()

    # MRF to be used
    offline_MRF_csv = CSV.read("Test/mrf_to_use.csv", DataFrame, header = false)
    offline_MRF_csv = Matrix(offline_MRF_csv)

    # Results directory
    test = "Test/MRF_usage"
    mkpath(test)
    
    # Directory where we store MRF information for each episode
    belief_info_dir = joinpath(test, "belief_info")
    mkpath(belief_info_dir)

    # File with the configurations of the rocks for each episode
    data_input=readdlm("Test/rocks_config.txt", '\t', Bool, '\n')

    num_rocks = 8
    

    for i = 1:100
        
        values_seen = fill(2.0, num_rocks)

        println("\nEpisode $i\n")
        pomdp = RockSamplePOMDP(map_size=(5,5),
			    rocks_positions=[(1,4), (2,1), (2,2), (3,2), (4,4), (5,5), (5,3), (2,5)],
                            sensor_efficiency=20.0,
                            options = Options(verbose=0, log_dir=test, episode=i, relations=1, n_rand_particles=2^num_rocks))


        solver = POMCPSolver(max_depth=70, c=20, tree_queries=100000, rng=MersenneTwister(987654321))
        planner = solve(solver, pomdp)

        init_state_file = data_input[i,1:num_rocks]

        if pomdp.options.relations == 0
            dist = initialstate(pomdp)
        elseif pomdp.options.relations == 1
            pomdp.mrf = offline_MRF_csv
            if pomdp.options.n_rand_particles == 2^num_rocks
	        dist = initialstate(pomdp)
	    else
                dist = initialstate(pomdp, pomdp.options.n_rand_particles, solver.rng)
	    end
        end


        upd=updater(planner)
        init_state = RSState(RSPos(1,1), SVector{num_rocks,Bool}(init_state_file), SVector{num_rocks,Bool}(falses(num_rocks))) 

        df_results = DataFrame(Position_s = SArray[], Rocks_s = SArray[], Rocks_sampled_s = SArray[], Action = Int64[], Reward = Float64[], Reward_disc = Float64[],  Position_sp = SArray[], Rocks_sp = SArray[], Rocks_sampled_sp = SArray[], Observation = Int64[], Num_particles = Int64[])
        
        for (s,a,r,sp,o,bp) in stepthrough(pomdp, planner, upd, dist, init_state, "s,a,r,sp,o,bp", max_steps=70, rng=MersenneTwister(987654321))
            num_particles = length(bp.particles)
            
            disc_rew = r * (pomdp.discount_factor ^ (pomdp.options.episode_step - 1))
            push!(df_results, (s.pos, s.rocks, s.collected, a, r, disc_rew, sp.pos, sp.rocks, sp.collected, o, num_particles))
            
            if a == 5 && in(s.pos, pomdp.rocks_positions)
                rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions)
                values_seen[rock_ind] = s.rocks[rock_ind]
            end


        end
        
        CSV.write("$test/output_episode_$i.csv", df_results)
        belief_count = joinpath(belief_info_dir, "belief_part_count_ep_$(pomdp.options.episode).csv")
        belief_prob = joinpath(belief_info_dir, "belief_part_prob_ep_$(pomdp.options.episode).csv")
        open(belief_count, "w") do io10
            writedlm(io10, pomdp.rocks_count_per_label_episode, ',') 
        end
        open(belief_prob, "w") do io11
            writedlm(io11, pomdp.rocks_prob_per_label_episode, ',') 
        end
    end

end

tempo = @time main()

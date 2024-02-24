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



# Update the matrices required to compute the MRF
function rock_matrix(n_rocks, rock_values, rocks_vals_mtx, count_occurences)
    n_r_seen = 0
    for j = 1:n_rocks
        if rock_values[j] != 2
            n_r_seen += 1
            if n_r_seen == 2
                break
            end
        end
    end

    if n_r_seen > 1
        for rock_a_id = 1:(n_rocks-1)
            r1_val = 0
            r2_val = 0
            if rock_values[rock_a_id] == 0
                r1_val = (rock_a_id*2)-1
            elseif rock_values[rock_a_id] == 1
                r1_val = rock_a_id*2
            else
                continue
            end
            for rock_b_id = (rock_a_id+1):n_rocks
                if rock_values[rock_b_id] == 0
                    r2_val = (rock_b_id*2)-1
                elseif rock_values[rock_b_id] == 1
                    r2_val = rock_b_id*2
                else
                    continue
                end
                rocks_vals_mtx[r1_val, r2_val] += 1
                count_occurences[(rock_a_id*2)-1, rock_b_id*2] += 1
                count_occurences[(rock_a_id*2)-1, (rock_b_id*2)-1] += 1
                count_occurences[rock_a_id*2, rock_b_id*2] += 1
                count_occurences[rock_a_id*2, (rock_b_id*2)-1] += 1
            end
        end
    end
end



# Main function of the MBL learning method
function main()
    
    # Results directory
    test = "Test/MBL_learning/"
    mkpath(test)
    
    # Directory where we store belief information for each episode
    belief_info_dir = joinpath(test, "belief_info")
    mkpath(belief_info_dir)

    # Directory where we store MRF information for each episode
    mrf_dir = joinpath(test, "mrf_info")
    mkpath(mrf_dir)

    # File with the configurations of the rocks for each episode
    data_input=readdlm("Test/rocks_config.txt", '\t', Bool, '\n')
    
    num_rocks = 8
    
    # Matrices required to update the MRF
    count_rocks_vals_mtx = zeros(Float32, 2*num_rocks, 2*num_rocks)
    count_rocks_seen = zeros(Int32, 2*num_rocks, 2*num_rocks)
    mrf_mtx = fill(0.25, (2*num_rocks,2*num_rocks))
    

    for i = 1:100
        
        values_seen = fill(2.0, num_rocks)

        println("\nEpisode $i\n")
        pomdp = RockSamplePOMDP(map_size=(5,5),
                            rocks_positions=[(1,4), (2,1), (2,2), (3,2), (4,4), (5,5), (5,3), (2,5)],
                            sensor_efficiency=20.0,
                            options = Options(verbose=0, log_dir=test, episode=i, relations=0))

    
        solver = POMCPSolver(max_depth=100, c=20, tree_queries=100000, rng=MersenneTwister(987654321))
        planner = solve(solver, pomdp)

        init_state_file = data_input[i,1:num_rocks]

        if pomdp.options.relations == 0
            dist = initialstate(pomdp)

        elseif pomdp.options.relations == 1
            if (i-1)%update_at == 0 
                mrf_mtx_tmp = deepcopy(mrf_mtx)
            end
            pomdp.mrf = mrf_mtx_tmp
            dist = initialstate(pomdp, pomdp.options.n_rand_particles, solver.rng)
        end


        upd=updater(planner)
        init_state = RSState(RSPos(1,1), SVector{num_rocks,Bool}(init_state_file), SVector{num_rocks,Bool}(falses(num_rocks)))

        df_results = DataFrame(Position_s = SArray[], Rocks_s = SArray[], Rocks_sampled_s = SArray[], Action = Int64[], Reward = Float64[], Reward_disc = Float64[],  Position_sp = SArray[], Rocks_sp = SArray[], Rocks_sampled_sp = SArray[], Observation = Int64[], Num_particles = Int64[])
        
        for (s,a,r,sp,o,bp) in stepthrough(pomdp, planner, upd, dist, init_state, "s,a,r,sp,o,bp", max_steps=100, rng=MersenneTwister(987654321))
            num_particles = length(bp.particles)

            disc_rew = r * (pomdp.discount_factor ^ (pomdp.options.episode_step - 1))
            push!(df_results, (s.pos, s.rocks, s.collected, a, r, disc_rew, sp.pos, sp.rocks, sp.collected, o, num_particles))
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

        # MRF update based on maximum likelihood state in the belief
        particle_idx = argmax(last(pomdp.rocks_prob_per_label_episode))-1
        values_seen = reverse(digits(particle_idx, base=2, pad=length(pomdp.rocks_positions)))
        rock_matrix(num_rocks, values_seen, count_rocks_vals_mtx, count_rocks_seen)
        
        for rock_a_id = 1:(length(pomdp.rocks_positions)-1)
            for rock_b_id = (rock_a_id+1):length(pomdp.rocks_positions)
                if count_rocks_seen[(rock_a_id*2)-1, (rock_b_id*2)-1] == 0
                    continue
                else
                    mrf_mtx[(rock_a_id*2)-1, (rock_b_id*2)-1] = count_rocks_vals_mtx[(rock_a_id*2)-1, (rock_b_id*2)-1] / count_rocks_seen[(rock_a_id*2)-1, (rock_b_id*2)-1]
                    mrf_mtx[(rock_a_id*2)-1, (rock_b_id*2)] = count_rocks_vals_mtx[(rock_a_id*2)-1, (rock_b_id*2)] / count_rocks_seen[(rock_a_id*2)-1, (rock_b_id*2)]
                    mrf_mtx[(rock_a_id*2), (rock_b_id*2)-1] = count_rocks_vals_mtx[(rock_a_id*2), (rock_b_id*2)-1] / count_rocks_seen[(rock_a_id*2), (rock_b_id*2)-1]
                    mrf_mtx[(rock_a_id*2), (rock_b_id*2)] = count_rocks_vals_mtx[(rock_a_id*2), (rock_b_id*2)] / count_rocks_seen[(rock_a_id*2), (rock_b_id*2)]
                end
            end
        end
        

        open("$mrf_dir/values_occurences_episode_$i.csv", "w") do io8
            writedlm(io8, count_rocks_vals_mtx, ',')
        end
        open("$mrf_dir/count_occurences_episode_$i.csv", "w") do io9
            writedlm(io9, count_rocks_seen, ',')
        end
        open("$mrf_dir/mrf_episode_$i.csv", "w") do io10
            writedlm(io10, mrf_mtx, ',') 
        end
    end

end

tempo = @time main()

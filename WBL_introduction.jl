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



# Compute an MRF considering all particles in the belief
function compute_mrfs_pf(probabilities_arr_pf, n_particles, num_rocks)
    mrf_pf = zeros(Float32, 2*num_rocks, 2*num_rocks)
    for p_idx in 1:n_particles   
        if probabilities_arr_pf[p_idx] > 0
            mrf_particle = zeros(Float32, 2*num_rocks, 2*num_rocks)
            particle = reverse(digits(p_idx, base=2, pad=num_rocks))
            for i in 1:(num_rocks-1)
                if particle[i] == 0
                    r1_idx = (i*2)-1
                else
                    r1_idx = i*2
                end
                for j in (i+1):num_rocks
                    if particle[j] == 0
                        r2_idx = (j*2)-1
                    else
                        r2_idx = j*2
                    end
                    mrf_particle[r1_idx, r2_idx] = probabilities_arr_pf[p_idx]
                end
            end
            mrf_pf = mrf_pf + mrf_particle
        end
    end
    return mrf_pf
end



# Main function of the WBL learning method. The learning process is stopped at a specific episode and then the learned MRF is used
function main()
    
    # Results directory
    test = "Test/WBL_introduction/"
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
    mrf_mtx = fill(0.25, (2*num_rocks,2*num_rocks))
    mrf_mtx_to_introduce = fill(0.25, (2*num_rocks,2*num_rocks)) 
    mrf_particle_filter = zeros(Float32, 2*num_rocks, 2*num_rocks)

    # Episode at which the learned MRF is introduced
    introduce_at = 40

    for i = 1:100
        
        values_seen = fill(2.0, num_rocks)

        println("\nEpisode $i\n")
        pomdp = RockSamplePOMDP(map_size=(5,5),
			    rocks_positions=[(1,4), (2,1), (2,2), (3,2), (4,4), (5,5), (5,3), (2,5)],
                            sensor_efficiency=20.0,
                            options = Options(verbose=0, log_dir=test, episode=i, relations=0, n_rand_particles=1000000))

    
        solver = POMCPSolver(max_depth=100, c=20, tree_queries=100000, rng=MersenneTwister(987654321))
        planner = solve(solver, pomdp)

        init_state_file = data_input[i,1:num_rocks]

	if i < introduce_at
	    println("MRF Learning\n")
            pomdp.options.relations = 0
            dist = initialstate(pomdp)
        elseif i == introduce_at
            println("Use of the MRF\n")
            pomdp.options.relations = 1
	    mrf_mtx_to_introduce = deepcopy(mrf_mtx)
            pomdp.mrf = mrf_mtx_to_introduce
            dist = initialstate(pomdp, pomdp.options.n_rand_particles, solver.rng)
	else
            println("Use of the MRF\n")
	    pomdp.options.relations = 1
	    pomdp.mrf = mrf_mtx_to_introduce
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
        
        # MRF update based on weighting each state in the belief by its likelihood
        mrf_particle_filter = compute_mrfs_pf(last(pomdp.rocks_prob_per_label_episode), 2^length(pomdp.rocks_positions), length(pomdp.rocks_positions))
        for rock_a_id = 1:(length(pomdp.rocks_positions)-1)
            for rock_b_id = (rock_a_id+1):length(pomdp.rocks_positions)
                mrf_mtx[(rock_a_id*2)-1, (rock_b_id*2)-1] = (mrf_mtx[(rock_a_id*2)-1, (rock_b_id*2)-1] * i + mrf_particle_filter[(rock_a_id*2)-1, (rock_b_id*2)-1]) / (i+1)
                mrf_mtx[(rock_a_id*2), (rock_b_id*2)-1] = (mrf_mtx[(rock_a_id*2), (rock_b_id*2)-1] * i + mrf_particle_filter[(rock_a_id*2), (rock_b_id*2)-1]) / (i+1)
                mrf_mtx[(rock_a_id*2)-1, (rock_b_id*2)] = (mrf_mtx[(rock_a_id*2)-1, (rock_b_id*2)] * i + mrf_particle_filter[(rock_a_id*2)-1, (rock_b_id*2)]) / (i+1)
                mrf_mtx[(rock_a_id*2), (rock_b_id*2)] = (mrf_mtx[(rock_a_id*2), (rock_b_id*2)] * i + mrf_particle_filter[(rock_a_id*2), (rock_b_id*2)]) / (i+1)
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
        
        open("$mrf_dir/mrf_episode_$i.csv", "w") do io10
            writedlm(io10, mrf_mtx, ',') 
        end
    end

end

tempo = @time main()

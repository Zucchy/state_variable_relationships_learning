# Compute the probability of each particle
function ComputeParticleProbability(random_particle::RSState, pf::UnweightedParticleFilter)

    num_rocks = length(pf.model.rocks_positions)

    particle_prob = 1.

    for rock_a_id = 1:(num_rocks-1)

        for rock_b_id = (rock_a_id+1):num_rocks
            r_a_val = random_particle.rocks[rock_a_id]

            r_b_val = random_particle.rocks[rock_b_id]
            
            full_potential_00_11 = pf.model.mrf[(rock_a_id*2)-1, (rock_b_id*2)-1] + pf.model.mrf[rock_a_id*2, rock_b_id*2]
            potential_00_11 = full_potential_00_11/2

            potential_01_10 = (1 - full_potential_00_11)/2

            if r_a_val == r_b_val
                potential = potential_00_11
            else
                potential = potential_01_10
            end
            particle_prob *= potential
        end
    end
    return particle_prob
end

function ParticleSelection(random_particles::Vector{RSState{K}}, pf::UnweightedParticleFilter) where K
    tree_queries = pf.n
    all_particles = []
    all_particles_prob = []
    sum_particles_prob = 0.
    for i=1:pf.model.options.n_rand_particles
        push!(all_particles, random_particles[i])
        particle_i_prob = ComputeParticleProbability(random_particles[i], pf)
        push!(all_particles_prob, particle_i_prob)
        sum_particles_prob += particle_i_prob

    end
    cumulative_particles_prob = zeros(Float64, 1, pf.model.options.n_rand_particles+1)

    for j=1:pf.model.options.n_rand_particles
        # utility vector for random sampling
        cumulative_particles_prob[j+1] = cumulative_particles_prob[j] + all_particles_prob[j]/sum_particles_prob
    end
    selected_particles = []
    for y=1:pf.n
        random_number = rand(pf.rng, Float64)
        particle_idx = 1
        incremental_prob = 0.
        while incremental_prob <= random_number
            particle_idx += 1
            incremental_prob = cumulative_particles_prob[particle_idx]
        end
        particle_idx-=1
        push!(selected_particles, all_particles[particle_idx])
    end
    return selected_particles
end

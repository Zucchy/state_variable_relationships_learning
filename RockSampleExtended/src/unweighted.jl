function particle_index(particle::RSState{K}) where K
    idx = 1
    for i=K:-1:1
        if particle.rocks[i]
            idx += 2^(K-i)
        end
    end
    return idx
end


# Count how many particles are in the belief by specifying the number of particles according to the configuration of the rocks (how many particles of type 00001, how many particles of type 00010,...)
function count_particles(up::UnweightedParticleFilter, b::ParticleCollection)
    n_rocks = length(up.model.rocks_positions)
    rocks_count_per_label = zeros(Int64, 2^n_rocks)
    all_particles_counter = length(b.particles)
    for part in b.particles
        p_idx = particle_index(part)
        rocks_count_per_label[p_idx] += 1
    end
    push!(up.model.rocks_count_per_label_episode, rocks_count_per_label)
    push!(up.model.rocks_prob_per_label_episode, rocks_count_per_label/all_particles_counter)
end

"""
UnweightedParticleFilter

A particle filter that does not use any reweighting, but only keeps particles if the observation matches the true observation exactly. This does not require obs_weight, but it will not work well in real-world situations.
"""
ParticleFilters.update(up::UnweightedParticleFilter, b::ParticleCollection, a, o) = update(up::UnweightedParticleFilter, b::ParticleCollection, a, o)

function update(up::UnweightedParticleFilter, b::ParticleCollection, a, o)
    new = Random.gentype(b)[]
        
    if up.model.options.episode_step == 2 # First step. Counter incremented in action_info before calling update.
        count_particles(up, b)
    end

    i = 1
    while i <= up.n
        particle_selected = []
        particle_rejected = []
        s = particle(b, mod1(i, n_particles(b)))
        sp, o_gen = @gen(:sp, :o)(up.model, s, a, up.rng)
        
        if o_gen == o
            push!(new, sp)
        end
        i += 1
    end
    
    count_particles(up, b)
  
    if isempty(new)
        warn("""
             Particle Depletion!

             The UnweightedParticleFilter generated no particles consistent with observation $o. Consider upgrading to a SIRParticleFilter or a BasicParticleFilter or creating your own domain-specific updater.
             """
            )
    end
    return ParticleCollection(new)
end


ParticleFilters.initialize_belief(up::UnweightedParticleFilter, b) = initialize_belief(up::UnweightedParticleFilter, b)
function initialize_belief(up::UnweightedParticleFilter, b)
    if up.model.options.relations == 0
        return ParticleCollection(collect(rand(up.rng, b) for i in 1:up.n))
    else
        MRF_selected_particles = ParticleSelection(b.vals, up)
        return ParticleCollection(MRF_selected_particles)
    end  
end

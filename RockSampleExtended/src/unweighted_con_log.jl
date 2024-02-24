"""
UnweightedParticleFilter

A particle filter that does not use any reweighting, but only keeps particles if the observation matches the true observation exactly. This does not require obs_weight, but it will not work well in real-world situations.
"""
ParticleFilters.update(up::UnweightedParticleFilter, b::ParticleCollection, a, o) = update(up::UnweightedParticleFilter, b::ParticleCollection, a, o)

function update(up::UnweightedParticleFilter, b::ParticleCollection, a, o)

    new = Random.gentype(b)[]
            
    i = 1
    while i <= up.n
        particle_selected = []
        particle_rejected = []
        s = particle(b, mod1(i, n_particles(b)))
        sp, o_gen = @gen(:sp, :o)(up.model, s, a, up.rng)
        if o_gen == o
            push!(new, sp)
            push!(particle_selected, s.pos)
            push!(particle_selected, s.rocks)
            push!(particle_selected, o)
            push!(particle_selected, sp.pos)
            push!(particle_selected, sp.rocks)
            push!(particle_selected, o_gen)
        else
            push!(particle_rejected, s.pos)
            push!(particle_rejected, s.rocks)
            push!(particle_rejected, o)
            push!(particle_rejected, sp.pos)
            push!(particle_rejected, sp.rocks)
            push!(particle_rejected, o_gen)
        end
        i += 1
    end
            
    if isempty(new)
        warn("""
             Particle Depletion!

             The UnweightedParticleFilter generated no particles consistent with observation $o. Consider upgrading to a SIRParticleFilter or a BasicParticleFilter or creating your own domain-specific updater.
             """
            )
    end
    return ParticleCollection(new)
end

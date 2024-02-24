function POMDPs.stateindex(pomdp::RockSamplePOMDP{K}, s::RSState{K}) where K
    if isterminal(pomdp, s)
        return length(pomdp)
    end
    rocks_ind = Int64.(s.rocks) .+ 1
    rocks_dim = fill(2, K)
    nx, ny = pomdp.map_size
    LinearIndices((nx, ny, rocks_dim...))[s.pos...,rocks_ind...]
end

function state_from_index(pomdp::RockSamplePOMDP{K}, si::Int) where K
    if si == length(pomdp)
        return pomdp.terminal_state
    end
    rocks_dim = fill(2, K)
    nx, ny = pomdp.map_size
    s = CartesianIndices((nx, ny, rocks_dim...))[si]
    pos = RSPos(s[1], s[2])
    rocks = SVector{K, Bool}([(s[i] - 1) for i=3:K+2])
    return RSState{K}(pos, rocks)
end

# the state space is the pomdp itself
POMDPs.states(pomdp::RockSamplePOMDP) = pomdp

Base.length(pomdp::RockSamplePOMDP) = pomdp.map_size[1]*pomdp.map_size[2]*2^length(pomdp.rocks_positions) + 1

# we define an iterator over it 
function Base.iterate(pomdp::RockSamplePOMDP, i::Int64=1)
    if i > length(pomdp)
        return nothing
    end
    s = state_from_index(pomdp, i)
    return (s, i+1)
end

function POMDPs.initialstate(pomdp::RockSamplePOMDP{K}) where K 
    probs = normalize!(ones(2^K), 1)
    states = Vector{RSState{K}}(undef, 2^K)
    for (i,rocks) in enumerate(Iterators.product(ntuple(x->[false, true], K)...))
        states[i] = RSState{K}(pomdp.init_pos, SVector(rocks), SVector{K, Bool}(falses(K)))
    end
    return SparseCat(states, probs)
end

function POMDPs.initialstate(pomdp::RockSamplePOMDP{K}, n_particles::Int, rng::AbstractRNG) where K
    probs = normalize!(ones(n_particles), 1)
    states = Vector{RSState{K}}(undef, n_particles)
    for i = 1:n_particles
        rocks = rand(rng, Bool, K)
        states[i] = RSState{K}(pomdp.init_pos, SVector{K,Bool}(rocks), SVector{K, Bool}(falses(K)))
    end
    return SparseCat(states, probs)
end

function POMDPs.transition(pomdp::RockSamplePOMDP{K}, s::RSState{K}, a::Int64) where K
    if isterminal(pomdp, s)
        return Deterministic(pomdp.terminal_state)
    end
    new_pos = next_position(s, a)
    if a == BASIC_ACTIONS_DICT[:sample] && in(s.pos, pomdp.rocks_positions)
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions)
        new_collected = MVector{K, Bool}(undef)
        for r=1:K
            new_collected[r] = r == rock_ind ? true : s.collected[r]
        end
        new_collected = SVector(new_collected)
    else 
        new_collected = s.collected
    end
    new_pos = RSPos(clamp(new_pos[1], 1, pomdp.map_size[1]), 
                    clamp(new_pos[2], 1, pomdp.map_size[2]))
    new_state = RSState{K}(new_pos, s.rocks, new_collected)
    return Deterministic(new_state)
end

function next_position(s::RSState, a::Int64)
    if a < N_BASIC_ACTIONS
        # the robot moves 
        return s.pos + ACTION_DIRS[a]
    elseif a >= N_BASIC_ACTIONS 
        # robot check rocks or samples
        return s.pos
    else
        throw("ROCKSAMPLE ERROR: action $a not valid")
    end
end

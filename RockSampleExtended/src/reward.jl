function POMDPs.reward(pomdp::RockSamplePOMDP, s::RSState, a::Int64)
    next_agent_pos = next_position(s, a)
    if next_agent_pos[1] > pomdp.map_size[1]
        #return pomdp.exit_reward
        return -100. # reward for illegal right exit
    elseif next_agent_pos[1] == 0 || next_agent_pos[2] == 0 || next_agent_pos[2] > pomdp.map_size[2]
        return -100.
    end

    if a == BASIC_ACTIONS_DICT[:sample] && in(s.pos, pomdp.rocks_positions)
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions)
        if s.collected[rock_ind] == true
            return -100.
        else
            return s.rocks[rock_ind] ? pomdp.good_rock_reward : pomdp.bad_rock_penalty 
        end
    elseif a == BASIC_ACTIONS_DICT[:sample] && !in(s.pos, pomdp.rocks_positions)
        return -100.
    end
    return 0.
end

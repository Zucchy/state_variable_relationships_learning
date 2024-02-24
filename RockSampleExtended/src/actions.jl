const N_BASIC_ACTIONS = 5
const BASIC_ACTIONS_DICT = Dict(:north => 1, 
                                :east => 2,
                                :south => 3,
                                :west => 4,
                                :sample => 5)

const ACTION_DIRS = (RSPos(0,1),
                    RSPos(1,0),
                    RSPos(0,-1),
                    RSPos(-1,0),
                    RSPos(0,0))

POMDPs.actions(pomdp::RockSamplePOMDP{K}) where K = 1:N_BASIC_ACTIONS+K
POMDPs.actionindex(pomdp::RockSamplePOMDP, a::Int64) = a

function POMDPs.actions(pomdp::RockSamplePOMDP{K}, s::RSState) where K
    if pomdp.options.knowledge == 0
        return 1:N_BASIC_ACTIONS+K
    elseif pomdp.options.knowledge == 1
        legal_actions = []
        for i in 1:N_BASIC_ACTIONS+K
            if check_legal(s, i, pomdp) == true
                push!(legal_actions, i)
            end
        end
        return legal_actions
    end
end

function check_legal(s::RSState, a, p::RockSamplePOMDP{K}) where K
    if a < N_BASIC_ACTIONS
        next_agent_pos = next_position(s, a)
        if next_agent_pos[1] > p.map_size[1]
            #return true # right exit allowed
            return false # right exit denied
        elseif next_agent_pos[1] == 0 || next_agent_pos[2] == 0 || next_agent_pos[2] > p.map_size[2]
            return false
        else
            return true
        end

    elseif a == N_BASIC_ACTIONS
        if in(s.pos, p.rocks_positions)
            rock_ind = findfirst(isequal(s.pos), p.rocks_positions)
            if s.collected[rock_ind] == false
                return true
            else
                return false
            end
        else 
            return false
        end
    elseif a > N_BASIC_ACTIONS
        rock_ind = a-N_BASIC_ACTIONS
        if s.collected[rock_ind] == false
            return true
        else
            return false
        end
    end
end

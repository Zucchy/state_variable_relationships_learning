BasicPOMCP.action_info(p::POMCPPlanner, b; tree_in_info=false) = action_info(p::POMCPPlanner, b; tree_in_info=false)

function action_info(p::POMCPPlanner, b; tree_in_info=false)
    local a::actiontype(p.problem)
    info = Dict{Symbol, Any}()
    try
        tree = POMCPTree(p.problem, b, p.solver.tree_queries)

        a = search(p, b, tree, info)
        p._tree = tree
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
        
    catch ex
        # Note: this might not be type stable, but it shouldn't matter too much here
        a = convert(actiontype(p.problem), default_action(p.solver.default_action, p.problem, b, ex))
        info[:exception] = ex
    end
    p.problem.options.episode_step += 1
    return a, info
end

BasicPOMCP.search(p::POMCPPlanner, b, t::POMCPTree, info::Dict) = search(p::POMCPPlanner, b, t::POMCPTree, info::Dict)

function search(p::POMCPPlanner, b, t::POMCPTree, info::Dict)
    all_terminal = true
    nquery = 0
    start_us = CPUtime_us()
    ucb_vals_progr = []

    progr_file = nothing
    if p.problem.options.verbose > 1

        progr_file = open(joinpath(p.problem.options.log_dir, "mcts_action_selection_progress_episode_$(p.problem.options.episode)", "step_$(p.problem.options.episode_step).csv"), "w")
    
        for a = POMDPs.actions(p.problem)
            write(progr_file, "act_$(a)_n,act_$(a)_v,act_$(a)_ucb")
        end
        write(progr_file, "\n")
    end
    mcts_desc_file = nothing
    if p.problem.options.verbose > 2
        mcts_desc_file = open(joinpath(p.problem.options.log_dir, "mcts_descent_details_episode_$(p.problem.options.episode)", "step_$(p.problem.options.episode_step).csv"), "w")
        write(mcts_desc_file, "descent,step,s_pos,s_rocks,s_collected,a,o,r,sp_pos,sp_rocks,sp_collected")
        for a = POMDPs.actions(p.problem)
            write(mcts_desc_file, "act_$(a)_n,act_$(a)_v,act_$(a)_ucb")
        end
        write(mcts_desc_file, "\n")
    end

    for i in 1:p.solver.tree_queries
        nquery += 1
        if CPUtime_us() - start_us >= 1e6*p.solver.max_time
            break
        end
        s = rand(p.rng, b)
        if !POMDPs.isterminal(p.problem, s)
            simulate(p, s, POMCPObsNode(t, 1), p.solver.max_depth, mcts_desc_file, i)
            all_terminal = false
        end
        
        ucb_vals_progr = []
        if p.problem.options.verbose  > 1
            ltn = log(t.total_n[1])
            for node in t.children[1]
                n = t.n[node]
                if n == 0 && ltn <= 0.0
                    criterion_value = t.v[node]
                elseif n == 0 && t.v[node] == -Inf
                    criterion_value = Inf
                else
                    criterion_value = t.v[node] + p.solver.c*sqrt(ltn/n)
                end

                push!(ucb_vals_progr, n)
                push!(ucb_vals_progr, t.v[node])
                push!(ucb_vals_progr, criterion_value)
            end
            writedlm(progr_file, transpose(ucb_vals_progr), ',') 
        end
    end
    if p.problem.options.verbose > 2
        close(mcts_desc_file)
    end
    if p.problem.options.verbose > 1
        close(progr_file)
    end

    info[:search_time_us] = CPUtime_us() - start_us
    info[:tree_queries] = nquery

    if all_terminal
        throw(AllSamplesTerminal(b))
    end

    h = 1
    ltn = log(t.total_n[h])
    best_node = first(t.children[h])
    best_v = t.v[best_node]
    @assert !isnan(best_v)
    ucb_vals = []
    for node in t.children[h]
        if t.v[node] >= best_v
            best_v = t.v[node]
            best_node = node
        end
        if p.problem.options.verbose == 1
            n = t.n[node]
            if n == 0 && ltn <= 0.0
                criterion_value = t.v[node]
            elseif n == 0 && t.v[node] == -Inf
                criterion_value = Inf
            else
                criterion_value = t.v[node] + p.solver.c*sqrt(ltn/n)
            end
            push!(ucb_vals, n)
            push!(ucb_vals, t.v[node])
            push!(ucb_vals, criterion_value)
        end
    end
    if p.problem.options.verbose > 1
        if p.problem.options.verbose == 2
            ucb_vals = ucb_vals_progr
        end
        writedlm(p.problem.options.mcts_act_sel_ptr, transpose(ucb_vals), ',')
        flush(p.problem.options.mcts_act_sel_ptr)
    end
    return t.a_labels[best_node]
end

function simulate(p::POMCPPlanner, s, hnode::POMCPObsNode, steps::Int, mcts_desc_file::Union{IOStream, Nothing}, descent::Int)
    if steps == 0 || isterminal(p.problem, s)
        return 0.0
    end

    t = hnode.tree
    h = hnode.node
    # Selection (MCTS)
    ltn = log(t.total_n[h])
    best_nodes = empty!(p._best_node_mem)
    best_criterion_val = -Inf

    n_array=[]
    v_array=[]
    ucb_array=[]

    for node in t.children[h]
        if p.problem.options.knowledge == 1 && !check_legal(s, t.a_labels[node], p.problem)
            t.n[node] = typemax(Int)
            t.v[node] = -Inf
            criterion_value = t.v[node]
            push!(n_array, t.n[node])
            push!(v_array, t.v[node])
            push!(ucb_array, criterion_value)
        else
            n = t.n[node]
            if n == 0 && ltn <= 0.0
                criterion_value = t.v[node]
            elseif n == 0 && t.v[node] == -Inf
                criterion_value = Inf
            else
                criterion_value = t.v[node] + p.solver.c*sqrt(ltn/n)
            end
            push!(n_array, n)
            push!(v_array, t.v[node])
            push!(ucb_array, criterion_value)

            if criterion_value > best_criterion_val
                best_criterion_val = criterion_value
                empty!(best_nodes)
                push!(best_nodes, node)
            elseif criterion_value == best_criterion_val
                push!(best_nodes, node)
            end
        end
    end

    ha = rand(p.rng, best_nodes)
    a = t.a_labels[ha]

    sp, o, r = @gen(:sp, :o, :r)(p.problem, s, a, p.rng)
    if p.problem.options.verbose > 2
        simul_info = []
        push!(simul_info, descent)
        push!(simul_info, p.solver.max_depth-steps)
        push!(simul_info, s.pos)
        push!(simul_info, s.rocks)
        push!(simul_info, s.collected)
        push!(simul_info, a)
        push!(simul_info, o)
        push!(simul_info, r)
        push!(simul_info, sp.pos)
        push!(simul_info, sp.rocks)
        push!(simul_info, sp.collected)
        for actn = POMDPs.actions(p.problem)
            push!(simul_info, n_array[actn])
            push!(simul_info, v_array[actn])
            push!(simul_info, ucb_array[actn])
        end
        writedlm(mcts_desc_file, transpose(simul_info), ',')
    end

    hao = get(t.o_lookup, (ha, o), 0)
    if hao == 0
        hao = insert_obs_node!(t, p.problem, ha, o) # EXPANSION (MCTS)
        v = estimate_value(p.solved_estimator, # ROLLOUT (MCTS)
                           p.problem,
                           sp,
                           POMCPObsNode(t, hao),
                           steps-1)
        R = r + discount(p.problem)*v
    else
        R = r + discount(p.problem)*simulate(p, sp, POMCPObsNode(t, hao), steps-1, mcts_desc_file, descent)
    end

    t.total_n[h] += 1 # BACKPROPAGATION (MCTS)
    t.n[ha] += 1
    t.v[ha] += (R-t.v[ha])/t.n[ha]

    return R
end



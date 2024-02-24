# RolloutSimulator 
POMDPSimulators.simulate(sim::RolloutSimulator, pomdp::POMDP, policy::Policy, updater::Updater, initial_belief, s) = simulate(sim::RolloutSimulator, pomdp::POMDP, policy::Policy, updater::Updater, initial_belief, s)

function simulate(sim::RolloutSimulator, pomdp::POMDP, policy::Policy, updater::Updater, initial_belief, s)
    
    if sim.eps == nothing
        eps = 0.0
    else
        eps = sim.eps
    end
    
    if sim.max_steps == nothing
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end

    disc = 1.0
    r_total = 0.0

    b = initialize_belief(updater, initial_belief)

    step = 1

    while disc > eps && !isterminal(pomdp, s) && step <= max_steps

        a = action(policy, s)

        sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a, sim.rng)

        r_total += disc*r

        s = sp

        bp = update(updater, b, a, o)
        b = bp

        disc *= discount(pomdp)
        step += 1
    end

    return r_total
end

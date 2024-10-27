num_agents = 50
settings = Dict("use_states_info_gain" => false)
parameters =Dict("lr_pA" => 1.0,"fr_pA" => 0.5, "alpha" => 1.0, "gamma" => 300.0)
ising_1D = Ising1D(num_agents, generative_model, settings, parameters);

T = 300

model_state_store = []
total_efe_1 = []
total_efe_2 = []

for t in 1:T
    model_state = update_model!(ising_1D)
    push!(model_state_store, deepcopy(model_state))
    
    # Sum efe_1 for each agent and push to total_efe_1
    efe_1 = sum([get_states(agent, "expected_free_energies")[1] for agent in values(ising_1D.agents)])
    push!(total_efe_1, efe_1)
    # Sum efe_2 for each agent and push to total_efe_2
    efe_2 = sum([get_states(agent, "expected_free_energies")[2] for agent in values(ising_1D.agents)])
    push!(total_efe_2, efe_2)

end

# Dynamic correlation
model_state_store_20th = [state[20] for state in model_state_store]
model_state_store_30th = [state[30] for state in model_state_store]

anim_correlation = @animate for t in 1:5:T
    p = plot(1:5:t, [mean(model_state_store_20th[i:min(i+4, end)]) for i in 1:5:t],
        label="20th Agent",
        xlabel="Time",
        ylabel="State", 
        title="Agent pair state correlation γ = 300", 
        xlims=(0, 300),
        ylims=(-0.1, 1.1),
        lw=2,
        color=:firebrick
    )
    plot!(p, 1:5:t, [mean(model_state_store_30th[i:min(i+4, end)]) for i in 1:5:t],
        label="30th Agent",
        lw=2,
        color=:springgreen
    )
end

gif(anim_correlation, "animations/agent_pair_correlation.gif", fps=8)



anim = @animate for (i, state) in enumerate(model_state_store)
    # Create the heatmap
    hm = heatmap(
        reshape(state, 1, :),
        title = "1D Lattice - $(length(ising_1D.agents)) Agents, t=$i, γ=300",
        xlabel = "",
        ylabel = "",
        ytick = false,
        xtick = false,
        legend = false,
        aspect_ratio = :equal,
        color = [:firebrick1, :turquoise3],
        titlefontsize = 10,
        size = (800, 50)
    )
    
    for (j, val) in enumerate(state)
        text_color = val == 0 ? "white" : "black"
        annotate!(hm, j, 1, text(string(val), :center, 8, text_color))
    end
end

gif(anim, "animations/ising1d.gif", fps=8)

anim_efe = @animate for t in 1:T
    p = plot(total_efe_1[1:t], color=:lightblue, legend=:topleft, label="(π) Active", title="Total Expected Free Energy, γ=300", size=(800, 400), xlims=(0, 300), ylims=(-140, -75))
    plot!(p, total_efe_2[1:t], color=:red, label="(π) Inactive")
end

gif(anim_efe, "animations/total_efe.gif", fps=8)

# Calculate the Hamiltonian energy.
# We use the following formula for the 1D Ising model:
# H = -sum(s_i * s_(i+1)) for all i
# where s_i are the spins (converted from 0/1 to -1/+1).
energy_vector = [
    -sum((2 .* state .- 1)[1:end-1] .* (2 .* state .- 1)[2:end])
    for state in model_state_store
]

anim_energy = @animate for t in 1:T
    plot(energy_vector[1:t], lw=2, color=:lawngreen, title="Hamiltonian Energy, γ=300", legend=false, xlims=(0, 300))
end

gif(anim_energy, "animations/hamiltonian_energy.gif", fps=8)



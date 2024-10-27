function update_model!(model::Ising1D)
    num_agents = length(model.model_state)
    next_state = similar(model.model_state)

    # Update each agent based on its neighbors' states and sample actions
    for i in 1:num_agents
        current_state = model.model_state[i]

        # Define neighbors without edge wrapping (edges observe missing neighbors as inactive)
        left_state = i > 1 ? model.model_state[i - 1] : 0    # Left neighbor is inactive for the first agent
        right_state = i < num_agents ? model.model_state[i + 1] : 0  # Right neighbor is inactive for the last agent

        # Compute observation using the formula: 1 - current_state * 4 + 1 - left_state * 2 + 1 - right_state + 1
        observation = (1 - current_state) * 4 + (1 - left_state) * 2 + (1 - right_state) + 1

        # Infer the agent's state with the observation
        agent = model.agents["agent_$i"]
        infer_states!(agent, [observation])

        # Update transitions if action is defined
        if get_states(agent)["action"] !== missing
            QS_prev = get_history(agent)["posterior_states"][end-1]
            update_B!(agent, QS_prev)
        end

        infer_policies!(agent)  # Infer the agent's policy

        # Sample the agent's action
        action = sample_action!(agent)[1]

        # Determine the next state for the agent based on the action
        next_state[i] = action == 1 ? 1 : 0  # 1=active, 0=inactive
    end

    # Update the model's state
    model.model_state .= next_state

    return model.model_state  # Return the updated model state
end
mutable struct Ising1D
    agents::Dict{String, ActiveInference.AIF} 
    generative_model::Dict 
    settings::Union{Dict, Nothing} 
    parameters::Union{Dict, Nothing}
    model_state::Vector{Int}  #  model's joint state

    # Constructor for Ising1D
    function Ising1D(num_agents::Int, generative_model::Dict, 
                     settings::Union{Dict, Nothing}=nothing, 
                     parameters::Union{Dict, Nothing}=nothing)

        agents = Dict{String, ActiveInference.AIF}()
        
        # Initialize agents
        for i in 1:num_agents
            agents["agent_$i"] = init_aif(
                generative_model[:A], 
                generative_model[:B]; 
                C=generative_model[:C], 
                pB=generative_model[:pB],
                settings=settings,
                parameters=parameters,
                verbose=false
            )
        end

        # Initialize model_state with random active/inactive states
        model_state = rand(0:1, num_agents)  # Random states (0=inactive, 1=active)

        println("Ising1D initialized with $num_agents agents")
        println("- Current model state is: $model_state")

        # Return the initialized struct instance
        new(agents, generative_model, settings, parameters, model_state)
    end
end




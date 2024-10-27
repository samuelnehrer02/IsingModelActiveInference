using Pkg
Pkg.activate("ising_model")
using ActiveInference
using LinearAlgebra
using Plots
theme(:juno)
using Serialization

# Lets start with a single line of agents for simplicity
# Each agent has two neighbors, one on each side
# Each agent can be in one of two states, active 1 or inactive 2
# Each agent observes the state of its neighbors in relation to its own state

# Model of the individual agent
# 1 = I am active, left is active, right is active 
# 2 = I am active, left is active, right is inactive 
# 3 = I am active, left is inactive, right is active 
# 4 = I am active, left is inactive, right is inactive  

# 5 = I am inactive, left is active, right is active 
# 6 = I am inactive, left is active, right is inactive 
# 7 = I am inactive, left is inactive, right is active  
# 8 = I am inactive, left is inactive, right is inactive 


states = [8]
observations = [8]
controls = [2]
policy_length = 1
A,B,C = create_matrix_templates(states, observations, controls, policy_length, "zeros")

A[1] .= I(8)

B[1][1:4, :, 1] .= 0.25
B[1][5:8, :, 2]  .= 0.25

B[1]

pB = deepcopy(B) * 4
C[1] = [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0]

generative_model = Dict(:A=>A, :B=>B, :C=>C, :pB=>pB);


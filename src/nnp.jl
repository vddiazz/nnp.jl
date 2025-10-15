module nnp

using JLD2
using LinearAlgebra
using NPZ
using Plots
using SpecialFunctions

#include("sampling.jl") # PENDING: USE PRE-GENERATED DATA WITH PYTHON
#export make_r
#export make_Q
#export make_grid

include("prod.jl")
export field_value
export field_grid
export make_field

end

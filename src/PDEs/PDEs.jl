abstract type AbstractPartialDifferentialEquation end

include("poisson.jl")
include("heat.jl")
include("temperature_correction.jl")
include("heat_timestep.jl")
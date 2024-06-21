module GFDM
using Reexport

@reexport using LinearAlgebra
@reexport using SparseArrays
@reexport using Printf

import Random
import Dates

@reexport using StaticArrays
@reexport using StructArrays
@reexport using DataFrames
@reexport using TypedTables

@reexport using NearestNeighbors
@reexport using WriteVTK
@reexport using JLD2

ENV["LC_NUMERIC"] = "C"
@reexport import Triangulate
@reexport import IterativeSolvers
@reexport import IncompleteLU

@reexport using CSV
@reexport using UnPack

@reexport import Optim

@reexport using SpecialFunctions

include("PointCloud/PointClouds.jl")

include("Utils/matrix_utils.jl")
include("Utils/manifoldpoints.jl")
include("Utils/roots.jl")
include("Utils/discrete_functions.jl")

include("IterativeSolvers/IterativeSolvers.jl")

include("Optimization/diagonaldominance.jl")
include("Optimization/leastsquares.jl")
include("Optimization/localmatrices.jl")

include("Boundary/boundaryflag.jl")
include("Boundary/interfaces.jl")

include("Operators/Operators.jl")

include("PDEs/PDEs.jl")

include("Interfaces/meshfree/MeshfreeInterface.jl")

include("TestCases/TestCases.jl")


ENV["LC_NUMERIC"] = "C"

# !Debug: This is only for development and should be deleted later
for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
        @eval export $n
    end
end

end

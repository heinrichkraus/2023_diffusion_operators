Base.@kwdef struct LinearSolver_gmres <: LinearSolverType

end

function solve_linear_system!(x, A, b, ::LinearSolver_gmres)
    IterativeSolvers.gmres!(x, A, b)
end

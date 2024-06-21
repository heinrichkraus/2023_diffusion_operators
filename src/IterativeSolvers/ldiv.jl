struct LinearSolver_ldiv <: LinearSolverType end

function solve_linear_system!(x, A, b, ::LinearSolver_ldiv)
    ldiv!(x, factorize(A), b)
end

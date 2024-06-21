function initialize_integrations(p, t, integration_statements; kwargs...)
    y = evaluate_integration(p, t, integration_statements; kwargs...)
    return DataFrame([y])
end

function evaluate_integration(p, t, integration_statements; kwargs...)
    integrations = NamedTuple{keys(integration_statements)}(
        map(f -> f(p, t), values(integration_statements))
    )
    return merge(integrations, merge(NamedTuple(kwargs)))
end

function evaluate_integrations!(df::DataFrame, p, t, integration_statements; kwargs...)
    y = evaluate_integration(p, t, integration_statements; kwargs...)
    push!(df, y, promote = true, cols = :union)
end
function boundary_matrix(pc, bnd_flag, neumann_conservative)
    M = zero(pc.sparsity)
    boundary_matrix!(M, pc, bnd_flag, neumann_conservative)
    return M
end

function parse_boundary_condition_base!(
    B,                                      # boundary condition matrix
    b,                                      # boundary condition rhs
    label::Symbol,                          # where to apply boundary condition (spatially)
    bcon::BoundaryCondition,                # boundary condition for dispatch
    pc::PointCloud
)
    # find all points belonging to label
    plist = findall(==(label), pc.p.label)
    for i = plist
        parse_boundary_condition!(B, b, bcon, i, pc)
    end

end

function parse_boundary_condition!(
    B,                                          # boundary condition matrix
    b,                                          # boundary condition rhs
    bcon::BoundaryCondition{F, <:Dirichlet},    # dirichlet boundary condition
    i,                                          # point index
    pc::PointCloud
) where F
    B[i,i] = 1
    b[i]   = bcon(pc.p[i])
end

function parse_boundary_condition!(
    B,                                      # boundary condition matrix
    b,                                      # boundary condition rhs
    bcon::BoundaryCondition{F, <:Neumann},  # Neumann boundary condition
    i,                                      # point index
    pc::PointCloud
) where F
    B[i, pc] = -directional_derivative_row(
        pc.p.n[i],
        pc.p.x[i],
        pc.p.h[i],
        pc.p.neighbors[i],
        bcon.bc.order,
        pc,
        bcon.bc.diagonal_dominance)
    b[i] = -bcon(pc.p[i])
end

function parse_boundary_condition!(
    B,                                                  # boundary condition matrix
    b,                                                  # boundary condition rhs
    bcon::BoundaryCondition{F, <:ConservativeNeumann},  # boundary function (point-dependent)
    i,                                                  # point index
    pc::PointCloud
) where F
    si, area = surface_volume_voronoi(i, pc)
    B[i, pc] = -neumann_weak(i, pc)
    b[i]     = pc.p.q[i] + pc.p.λ[i] * si / area * bcon(pc.p[i])
end

function parse_boundary_condition!(
    B, b,
    bcon::BoundaryCondition{F, <:Jump},
    i,
    pc::PointCloud
) where F
    j = pc.p.partner[i]
    B[i, i] = 1
    B[i, j] = -1
    b[i]    = bcon(pc.p[i])
end

function parse_boundary_condition!(
    B, b,
    bcon::BoundaryCondition{F, <:FluxJump},
    i,
    pc::PointCloud
) where F
    j = pc.p.partner[i]
    for k = pc.p.neighbors[i]
        B[i, k] = 0
    end
    add_to_matrix_row!(B, pc.p.λ[j] * directional_derivative_row(pc.p.n[j], pc.p.x[j], pc.p.h[j], pc.p.neighbors[j], bcon.order, pc, bcon.diagonal_dominance), i, pc.p.neighbors[j])
    add_to_matrix_row!(B, pc.p.λ[i] * directional_derivative_row(pc.p.n[i], pc.p.x[i], pc.p.h[i], pc.p.neighbors[i], bcon.order, pc, bcon.diagonal_dominance), i, pc.p.neighbors[i])
    b[i] = bcon(pc.p[i])
end
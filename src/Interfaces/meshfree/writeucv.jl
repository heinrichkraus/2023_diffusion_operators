function writeucv(meshpath::String, h::Number, bnd_name::String)
    io = open("USER_common_variables.dat", "w")
    write(io, """
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # c  Solver Section
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    KOP(1) =  LIQUID EULERIMPL V:IMPLICIT T:NONE  v--

    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # c  Saving Section
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    SAVE_format(1) = 'ASCII N---'
    SAVE_first(1)    = 1
    SAVE_interval(1) = 1
    SAVE_file = 'pointcloud'
    SAVE_path = 'ascii'
    SAVE_ITEM = (%SAVE_scalar%, [-Y%ind_n(1)%], "nx" )
    SAVE_ITEM = (%SAVE_scalar%, [-Y%ind_n(2)%], "ny" )
    SAVE_ITEM = (%SAVE_scalar%, [-Y%ind_n(3)%], "nz" )
    SAVE_ITEM = (%SAVE_scalar%, [Y%ind_h%],    "h"  )
    SAVE_ITEM = (%SAVE_scalar%, [Y%ind_BC%],   "BC" )
    SAVE_ITEM = (%SAVE_scalar%, [Y%ind_kob%],  "kob")

    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # c  Time Step Control Section
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    DELT_dt_variable = 0
    DELT_dt_start    = 0.001
    DELT_dt          = 1.0
    Tstart           = 0
    Tend             = 10000
    TimeIntegration_N_final = 5

    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # c  Smoothing Length Section
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    USER_h_funct = 'CONS'
    USER_h_max = $h
    USER_h_min = $h
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # c   Boundary Elements Section
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    begin_boundary_elements{}
        include{$meshpath}
    end_boundary_elements

    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # c   Physical Properties Section
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    density(\$MatUSER\$)    = 1000
    cv(\$MatUSER\$)         = 1500
    lambda(\$MatUSER\$)    = 2
    eta(\$MatUSER\$)        = 5
    mue(\$MatUSER\$)        = 0
    sigma(\$MatUSER\$)      = 0
    heatsource(\$MatUSER\$) = 0
    gravity(\$MatUSER\$)    = (0.0, 0.0, 0.0)

    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # c   Move Section
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    MOVE(\$NO_MOVE\$) = ( %MOVE_velocity%, 0.0, 0.0, 0 )

    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # c   Active Section
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    ACTIVE(\$init_always\$)   = ( %ACTIVE_init%, %ACTIVE_always% )

    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # c   Alias Section
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    begin_alias{"BoundaryElements"}
        "$bnd_name" = "BC\$wall\$ ACTIVE\$init_always\$ IDENT%IDENT_wall%  MAT\$MatUSER\$  TOUCH%TOUCH_always%  MOVE\$NO_MOVE\$ CHAMBER1 "
    end_alias
    """)
    close(io)
end

function writecv(dim::Integer)
    @assert 2 <= dim <= 3
    io = open("common_variables.dat", "w")
    write(io, "nue = $dim")
    close(io)
end
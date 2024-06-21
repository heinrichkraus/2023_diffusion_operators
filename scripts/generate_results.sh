#!/bin/bash

julia -t auto poisson_smooth.jl
julia -t auto poisson_smooth_difficult.jl
julia -t auto poisson_discontinuous.jl
julia -t auto heat_smooth.jl
julia -t auto heat_discontinuous.jl

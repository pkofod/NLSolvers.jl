using NLSolvers
using Test
using StaticArrays
#using Optim
#using LineSearches
using Printf
using LinearAlgebra: norm, I

include("noalloc_static.jl")
include("scalar/runtests.jl")
include("optimize/runtests.jl")
include("specialtypes/runtests.jl")
include("newton.jl")

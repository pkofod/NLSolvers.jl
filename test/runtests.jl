using NLSolvers
using Test
using StaticArrays
#using Optim
#using LineSearches
using Printf
using LinearAlgebra: norm, I
import Random
Random.seed!(41234)
include("problems.jl")
include("optimize/runtests.jl")
include("noalloc_static.jl")
include("scalar/runtests.jl")
include("nlsolve/runtests.jl")
include("fixedpoints/runtests.jl")
include("specialtypes/runtests.jl")
include("newton.jl")
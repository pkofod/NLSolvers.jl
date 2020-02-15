# I got 99 ...
problems = Dict()
problems["unconstrained"] = Dict()

include("unconstrained/from_optim.jl")
include("unconstrained/twicediffed.jl")

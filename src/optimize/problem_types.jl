"""
  MinProblem(...)
A MinProblem (Minimization Problem), is used to represent the mathematical problem
of finding local minima of the given objective function. The problem is defined by `objective`
which is an appropriate objective type (for example `NonDiffed`, `OnceDiffed`, ...)
for the types of algorithm to be used. The constraints of the problem are encoded
in `constraints`. See the documentation for supported types of constraints
including convex sets, and more. It is possible to explicitly state that there
are bounds constraints and manifold constraints on the inputs.

Options are stored in `options` and must be an appropriate options type. See more information
about options using `?MinOptions`.
"""
struct MinProblem{O<:ObjWrapper, B, M, C, Opt}
    objective::O
    bounds::B
    manifold::M
    constraints::C
    options::Opt
end
MinProblem(obj=nothing, bounds=nothing, manifold=nothing, constraints=nothing, options=nothing)

#MinOptions

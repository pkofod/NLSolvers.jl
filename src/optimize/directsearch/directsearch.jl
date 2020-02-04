include("neldermead.jl")
solve!(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, vs::ValuedSimplex) =
  minimize!(prob.objective, vs, NelderMead())

solve!(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, vs::ValuedSimplex, solver::NelderMead) =
  minimize!(prob.objective, vs, solver)

solve!(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, x0, solver::NelderMead) =
  minimize!(prob.objective, x0, solver)
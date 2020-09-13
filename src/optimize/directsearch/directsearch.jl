include("neldermead.jl")
solve!(prob::OptimizationProblem{<:Any, <:Nothing, <:Euclidean, <:Nothing}, vs::ValuedSimplex) =
  minimize!(prob.objective, vs, NelderMead(), MinOptions())

solve!(prob::OptimizationProblem{<:Any, <:Nothing, <:Euclidean, <:Nothing}, vs::ValuedSimplex, solver::NelderMead) =
  minimize!(prob.objective, vs, solver, MinOptions())

solve!(prob::OptimizationProblem{<:Any, <:Nothing, <:Euclidean, <:Nothing}, x0, solver::NelderMead) =
  minimize!(prob.objective, x0, solver, MinOptions())
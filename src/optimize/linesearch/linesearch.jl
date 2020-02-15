"""
  LineSearch(scheme, linesearcher)
"""
struct LineSearch{S, LS}
  scheme::S
  linesearcher::LS
end
LineSearch() = LineSearch(DBFGS(), Backtracking())
LineSearch(m) = LineSearch(m, Backtracking())

hasprecon(ls::LineSearch) = hasprecon(modelscheme(ls))
summary(ls::LineSearch) = summary(modelscheme(ls))*" with "*summary(algorithm(ls))

function initial_preconditioner(approach::LineSearch, x)
  method = modelscheme(approach)
  initial_preconditioner(method, x, hasprecon(method))
end

modelscheme(ls::LineSearch) = ls.scheme
algorithm(ls::LineSearch) = ls.linesearcher
include("conjugategradient.jl")
export ConjugateGradient

include("quasinewton.jl")
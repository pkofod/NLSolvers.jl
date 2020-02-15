struct NoPrecon end
struct HasPrecon end
function initial_preconditioner(method, x, ::NoPrecon)
  nothing
end
function initial_preconditioner(method, x, ::HasPrecon)
  method.P(x)
end
update_preconditioner(method, x, P) = update_preconditioner(method, x, P, hasprecon(method))
update_preconditioner(method, x, P, ::NoPrecon) = nothing
update_preconditioner(method, x, P, ::HasPrecon) = method.P(x, P)
precondition(d, ::Nothing, ∇f) = copyto!(d, ∇f)
precondition(d, P, ∇f) = ldiv!(d, P, ∇f)
precondition(::Nothing, ∇f) = ∇f
precondition(P, ∇f) = P\∇f
# The simple
struct Newton{T1, Tlin} <: QuasiNewton{T1}
   approx::T1
   linsolve::Tlin
end
hasprecon(::Newton) = NoPrecon()
# struct DefaultSequence end
DefaultNewtonLinsolve(B::Number, g) = B\g
function DefaultNewtonLinsolve(B, g)
  B\g
end
function DefaultNewtonLinsolve(d, B, g)
  d .= (B\g)
end
Newton(;approx=Direct(), linsolve=DefaultNewtonLinsolve) = Newton(approx, linsolve)
summary(::Newton{<:Direct, <:typeof(DefaultNewtonLinsolve)}) = "Newton's method with default linsolve"
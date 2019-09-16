# The simple
struct Newton{T1, Tlin, TSeq} <: QuasiNewton{T1}
   approx::T1
   linsolve::Tlin
   sequence::TSeq
end
# struct DefaultSequence end
Newton(approx) = Newton(approx, \, Backtracking())
Newton(;approx=Direct(), linsolve=\, sequence=NWI()) = Newton(approx, sequence)

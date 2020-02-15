# The simple
struct Newton{T1, Tlin, TSeq} <: QuasiNewton{T1}
   approx::T1
   linsolve::Tlin
   sequence::TSeq
end
hasprecon(::Newton) = NoPrecon()

# struct DefaultSequence end
Newton(;approx=Direct(), linsolve=\, sequence=NWI()) = Newton(approx, linsolve, sequence)

# The simple
struct Newton{T1, TSeq} <: QuasiNewton{T1}
   approx::T1
   sequence::TSeq
end
# struct DefaultSequence end
Newton(approx) = Newton(approx, BackTracking())
Newton(;approx=Direct(), sequence=NWI()) = Newton(approx, sequence)

export Newton

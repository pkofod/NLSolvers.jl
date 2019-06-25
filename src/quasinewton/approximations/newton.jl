# The simple
struct Newton{T1, TSeq} <: QuasiNewton{T1}
   approx::T1
   sequence::TSeq
end
struct DefaultSequence end
Newton(approx) = Newton(approx, DefaultSequence())
Newton(;approx = Direct(), sequence=DefaultSequence()) = Newton(approx, sequence)

export Newton

struct GradientDescent{T1} <: QuasiNewton{T1}
    approx::T1
end
GradientDescent() = GradientDescent(Direct())
update!(scheme::GradientDescent, A, s, y) = A
update(scheme::GradientDescent, A, s, y) = A

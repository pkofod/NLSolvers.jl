struct GradientDescent{T1} <: QuasiNewton{T1}
    approx::T1
end
update!(scheme::GradientDescent{<:Inverse}, A::UniformScaling, s, y) = A
update(scheme::GradientDescent{<:Inverse}, A::UniformScaling, s, y) = A
update!(scheme::GradientDescent{<:Direct}, A, s, y) = A
update(scheme::GradientDescent{<:Direct}, A, s, y) = A

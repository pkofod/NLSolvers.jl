function find_direction(B, P, ∇f, scheme::QuasiNewton{<:Direct})
   return -(B\∇f)
end
function find_direction(A, P, ∇f, scheme::QuasiNewton{<:Inverse})
   return -A*∇f
end
function find_direction!(d, B, P,∇f, scheme::QuasiNewton{<:Direct})
   d .= -(B\∇f)
   d
end
function find_direction!(d, A, P, ∇f, scheme::QuasiNewton{<:Inverse})
   rmul!(mul!(d, A, ∇f), -1)
   d
end
function find_direction(B, P, ∇f, scheme::GradientDescent)
   -precondition(P, ∇f)
end
function find_direction!(d, B, P,∇f, scheme::GradientDescent)
   d = precondition(d, P, ∇f)
   d .= .-d
end

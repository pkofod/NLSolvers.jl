struct ASProjNewton
end

## Update the active set here
function active_model(prob::MinProblem{<:Any, <:Any, <:Nothing, <:Nothing}, an::ActiveNewton, B, ∇fx, x)
  # split up the approach into the hessian approximation scheme and line search
  epsg = sqrt(eps(eltype(x)))
  m = _manifold(objective)
  lower, upper = lowerbounds(prob), upperbounds(prob)
  # Binding set 1
  # true if free
  lower_e = lower .+ epsg
  upper_e = upper .- epsg
  sl = .!( ((x .<= lower_e) .& (∇fx .>= 0)) .| ((x .>= upper_e) .& (∇fx .<= 0)) )
  Ix = initialh(x) # An identity matrix of correct type

  binding = isbinding.(sl, sl')
  ## The hessian needs to be adapted to ignore the inactive region
  Hhat = cdiag.(B, binding, Ix)
end

## The gradient needs to be zeroed out for the inactive region
# set jxc to 0 if var in binding set

∇fxc = ∇fx .* sl
end
function cdiag(x, c, i)
    if c
        return x
    else
        if i == 1.
            return 1.
        else
            return 0.
        end
    end
end
isbinding(i, j) = i & j

function iterate(::InPlace, qnvars, objvars, P, approach, prob, obj, options, is_first=nothing)
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    Tf = typeof(fx)
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    # Move nexts into currs
    fx = fz
    x = copy(z)
    ∇fx = copy(∇fz)

    # Diagonal B wrt the active set A
    B, A = active_model!(prob, B, ∇fx, x)

    # Update preconditioner
    P = update_preconditioner(approach, x, P)
    # Update current gradient and calculate the search direction
    d = find_direction(B, P, ∇fx, scheme) # solve Bd = -∇fx
    φ = _lineobjective(mstyle, prob, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1))

    # Update current gradient and calculate the search direction
    d = find_direction(Hhat, ∇fxc, scheme) # solve Bd = -∇fx
    φ = LineObjective(objective, x, d, fx, dot(∇fxc, d))
    # # Perform line search along d

    # Should the line search project at each step?
    α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1))

    # # Calculate final step vector and update the state
    s = @. α * d
    z = @. x + s
    if isboundedonly(prob)
        z = clamp.(z, lower, upper)
        s = @. z - x
    end
    
    # Update approximation
    fz, ∇fz, B, s, y = update_obj(obj, s, ∇fx, z, ∇fz, B, scheme, is_first)
    ∇fzc = ∇fz .* sl

    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=B, Pg=Pg), P, QNVars(d, s, y) 
end
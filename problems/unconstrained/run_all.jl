# run_all for unconstrained
function run_all(obj, x0, approx, globals)
    res = []
    _res = minimize!(obj, copy(x0), approx)
    push!(res, _res)
    for glob in globals
      _res = minimize!(obj, copy(x0), (approx, glob))
      printstyled(_res; color=maximum(abs, _res[3]) ≤ sqrt(eps(Float64)) ? :green : :red);println()
      push!(res, _res)
    end
    res
end

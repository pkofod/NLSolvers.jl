function solve!(prob::NEqProblem, x, method::Anderson, options=NEqOptions())
  fixedpoint!((x, F)->prob.R(x, F).+x, x, method;
                     # kwargs
                     Gx = similar(x),
                     Fx = similar(x),
                     f_abstol=sqrt(eps(eltype(x))),
                     maxiter=options.maxiter)

end

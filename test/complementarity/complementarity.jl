using NLsolve

M = [0  0 -1 -1 ;
     0  0  1 -2 ;
     1 -1  2 -2 ;
     1  2 -2  4 ]

q = [2; 2; -2; -6]

function f!(x, fvec)
    fvec = M * x + q
end


r = mcpsolve(f!, [0., 0., 0., 0.], [Inf, Inf, Inf, Inf],
             [1.25, 0., 0., 0.5], reformulation = :smooth, autodiff = true)

x = r.zero  # [1.25, 0.0, 0.0, 0.5]
@show dot( M*x + q, x )  # 0.5

sol = [2.8, 0.0, 0.8, 1.2]
@show dot( M*sol + q, sol )  # 0.0

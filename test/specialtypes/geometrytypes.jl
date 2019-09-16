using GeometryTypes, Test
function f(G, x)
    fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

    if !(G == nothing)
        G1 = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
        G2 = 200.0 * (x[2] - x[1]^2)
        gx = Point(G1,G2)

        return fx, gx
    else
        return fx
    end
end
f_obj = OnceDiffed(f)
minimize(f_obj, Point(3.0,3.0), GradientDescent(Inverse()), I, OptOptions())
minimize(f_obj, Point(3.0,3.0), BFGS(Inverse()), I, OptOptions())
minimize(f_obj, Point(3.0,3.0), DFP(Inverse()), I, OptOptions())
minimize(f_obj, Point(3.0,3.0), SR1(Inverse()), I, OptOptions())

minimize(f_obj, Point(3.0,3.0), GradientDescent(Direct()), I, OptOptions())
minimize(f_obj, Point(3.0,3.0), BFGS(Direct()), I, OptOptions())
minimize(f_obj, Point(3.0,3.0), DFP(Direct()), I, OptOptions())
minimize(f_obj, Point(3.0,3.0), SR1(Direct()), I, OptOptions())

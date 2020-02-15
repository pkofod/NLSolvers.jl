 @test sprint((io, t) -> show(io, MIME"text/plain"(), t), ft_normal) == """
    FittedPumasModel

    Successful minimization:                true

    Likelihood approximation:         Pumas.FOCE
    Deviance:                          8939.5062
    Total number of observation records:    4669
    Number of active observation records:   4669
    Number of subjects:                        7

    -----------------------
                 Estimate
    -----------------------
    tvcl          0.15977
    tvv           3.6998
    tvka          0.89359
    ec50         14.917
    gaeffect     10.952
    Ω₁,₁          0.071966
    Ω₂,₂          0.087925
    ν            23.217
    -----------------------
    """
    end
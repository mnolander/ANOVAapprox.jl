# ANOVAapprox.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://nfft.github.io/ANOVAapprox.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://nfft.github.io/ANOVAapprox.jl/dev)
[![ci](https://github.com/NFFT/ANOVAapprox.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/NFFT/ANOVAapprox.jl/actions?query=workflow%3ACI+branch%3Amain)
[![codecov](https://codecov.io/gh/NFFT/ANOVAapprox.jl/branch/main/graph/badge.svg?token=5RUDL3Z3S5)](https://codecov.io/gh/NFFT/ANOVAapprox.jl)
[![Aqua QA](https://img.shields.io/badge/Aqua.jl-%F0%9F%8C%A2-aqua.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package provides a framework for the method ANOVAapprox to approximate high-dimensional functions with a low superposition dimension or a sparse ANOVA decomposition from scattered data. The method has been dicussed and applied in the following articles/preprints:

 -  [**Approximation of high-dimensional periodic functions with Fourier-based methods** by D. Potts and M. Schmischke](https://doi.org/10.1137/20M1354921)
 - [**Learning multivariate functions with low-dimensional structures using polynomial bases** by D. Potts and M. Schmischke](https://doi.org/10.1016/j.cam.2021.113821)
 -  [**Interpretable approximation of high-dimensional data** by D. Potts and M. Schmischke](https://arxiv.org/abs/2103.13787)
 -  [**Interpretable transformed ANOVA approximation on the example of the prevention of forest fires** by D. Potts and M. Schmischke](https://arxiv.org/abs/2110.07353)


`ANOVAapprox.jl` provides the following functionality:
- approximation of high-dimensional periodic and nonperiodic functions with a sparse ANOVA decomposition
- analysis tools for interpretability (global sensitvitiy indices, attribute ranking)

## Getting started

In Julia you can get started by typing

```julia
] add ANOVAapprox
```

then checkout the [documentation](https://nfft.github.io/ANOVAapprox.jl/stable/).

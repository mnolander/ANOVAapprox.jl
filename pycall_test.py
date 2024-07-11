from juliacall import Main as jl
import juliapkg
import numpy as np

jl.seval("using Pkg")
jl.seval("Pkg.add(url=\"https://github.com/mnolander/ANOVAapprox.jl\")")
jl.seval("using ANOVAapprox")

d = 3
M = 10_000
X = np.random.rand(d, M)
y = np.random.rand(M)
U = 2
N = np.array([100, 10])

X = jl.convert(jl.Matrix, X)
y = jl.convert(jl.Vector, y)
N = jl.convert(jl.Vector, N)

approx_result = jl.ANOVAapprox.approx(X, y, U, N, "cos")

lmda = np.array([0.0, 1.0])

lmda_mat = np.random.rand(d, 1_000)
lmda_mat = jl.convert(jl.Matrix, lmda_mat)

lmda = jl.convert(jl.Vector, lmda)

jl.ANOVAapprox.approximate(approx_result, lmda=lmda)

eval_output = jl.ANOVAapprox.evaluate(approx_result, lmda_mat)
print("evaluate() output:", eval_output)

eval_anova_output = jl.ANOVAapprox.evaluateANOVAterms(approx_result, lmda_mat)
print("evaluateANOVAterms() output:", eval_anova_output)

eval_shap_output = jl.ANOVAapprox.evaluateSHAPterms(approx_result, lmda_mat)
print("evaluateSHAPterms() output:", eval_shap_output)
from juliacall import Main as jl
import juliapkg
import numpy as np

# juliapkg.add("ANOVAapprox", uuid="5e027bd6-ab01-4733-8320-e0223e929ebb")
juliapkg.project()

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

print("X type after conversion:", type(X))
print("y type after conversion:", type(y))
print("N type after conversion:", type(N))

approx_result = jl.ANOVAapprox.approx(X, y, U, N, "cos")

# lmda = np.array([0.0, 1.0])
lmda = 1.0

lmda = jl.convert(jl.Float64, lmda)

print("lmda type after conversion:", type(lmda))

jl.ANOVAapprox.approximate(approx_result, lmda)

# jl.seval("result = ANOVAapprox.get_orderDependentBW([[1, 2, 3], [4, 5, 6]], [3, 7, 1])")
# jl.seval("println(result)")
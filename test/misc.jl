d = 3
ds = 2
M = 10_000
bw = [100, 10]

X = rand(rng, d, M)
y = rand(M)

U = Vector{Vector{Int64}}(undef, 3)
U[1] = []
U[2] = [1]
U[3] = [1, 3]

bwU = ANOVAapprox.get_orderDependentBW(U, bw)

ads = ANOVAapprox.approx(X, y, ds, bw, "cos")
aU = ANOVAapprox.approx(X, y, U, bwU, "cos")
ANOVAapprox.approximate(aU, lambda = [0.0, 1.0], solver = "lsqr")
ANOVAapprox.evaluate(aU, X, 0.0)
ads = ANOVAapprox.approx(X, y, ds, bw, "std")
aU = ANOVAapprox.approx(X, y, U, bwU, "std")
ANOVAapprox.approximate(aU, lambda = [0.0, 1.0], solver = "lsqr")
ANOVAapprox.evaluate(aU, X, 0.0)
ANOVAapprox.get_AttributeRanking(aU)

try
    ANOVAapprox.approximate(aU, lambda = [0.0, 1.0], solver = "bananarama")
catch e
end


try
    ANOVAapprox.approx(X, complex(y), ds, bw, "cos")
catch e
end

try
    ANOVAapprox.approx(X, vcat(y, 1.0), ds, bw, "cos")
catch e
end

try
    ANOVAapprox.approx(X, vcat(y, 1.0), ds, vcat(bw, 8), "cos")
catch e
end

Xc = copy(X)
Xc[1, 1] = 7.0

try
    ANOVAapprox.approx(Xc, y, ds, bw, "cos")
catch e
end

try
    ANOVAapprox.approx(Xc, y, ds, bw, "cheb")
catch e
end

try
    ANOVAapprox.approx(Xc, complex(y), ds, bw, "per")
catch e
end

try
    ANOVAapprox.approx(Xc, y, ds, bw, "bananarama")
catch e
end

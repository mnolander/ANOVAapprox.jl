using ANOVAapprox
using Test
using Random 

include("TestFunction.jl")

using .TestFunction

rng = MersenneTwister(1234)

d = 6
ds = 2
M = 10_000
max_iter = 50
bw = [ 100, 10 ]
λs = [ 0.0, 1.0 ]

X = rand( rng, d, M ) .- 0.5
y = [ TestFunction.f(X[:,i]) for i = 1:M ]

f = ANOVAapprox.periodic_approx( X, complex(y), ds, bw; method = "lsqr" ) 

ANOVAapprox.approximate(f, lambda=λs, max_iter=max_iter)

bw = [ 128, 32 ]
f2 = ANOVAapprox.periodic_approx( X, complex(y), ds, bw; method = "lsqr", active_set=TestFunction.AS ) 

ANOVAapprox.approximate(f2, lambda=λs, max_iter=max_iter)

d = ANOVAapprox.get_L2error( f2, TestFunction.norm(), TestFunction.fc ) 

@test d[0.0] < 5*10^(-3)
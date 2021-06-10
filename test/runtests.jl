using ANOVAapprox
using Distributions
using Test
using Random 

include("TestFunctionPeriodic.jl")
include("TestFunctionNonPeriodic.jl")

using .TestFunctionPeriodic
using .TestFunctionNonPeriodic

rng = MersenneTwister(1234)

#### PERIODIC TEST ####

d = 6
ds = 2
M = 10_000
max_iter = 50
bw = [ 100, 10 ]
λs = [ 0.0, 1.0 ]

X = rand( rng, d, M ) .- 0.5
y = [ TestFunctionPeriodic.f(X[:,i]) for i = 1:M ]

a = ANOVAapprox.periodic_approx( X, complex(y), ds, bw; method = "lsqr" ) 
ANOVAapprox.approximate(a, lambda=λs, max_iter=max_iter)

bw = [ 128, 32 ]
a2 = ANOVAapprox.periodic_approx( X, complex(y), ds, bw; method = "lsqr", active_set=TestFunctionPeriodic.AS ) 

ANOVAapprox.approximate(a2, lambda=λs, max_iter=max_iter)
r = ANOVAapprox.get_AttributeRanking( a2, 0.0 )
d = ANOVAapprox.get_L2error( a2, TestFunctionPeriodic.norm(), TestFunctionPeriodic.fc ) 
println( ANOVAapprox.get_l2error(a2) )

@test d[0.0] < 5*10^(-3)

#### NONPERIODIC TEST ####

d = 8
ds = 2
M = 100_000
max_iter = 50
bw = [ 20, 4 ]
λs = [ 0.0, 1.0 ]

(X, y) = TestFunctionNonPeriodic.generateData( M, false, rng )

a = ANOVAapprox.nperiodic_approx( X, complex(y), ds, bw; basis="cheb", method="fista" )
ANOVAapprox.approximate(a, lambda=λs, max_iter=max_iter, smoothness=1.0)

a2 = ANOVAapprox.nperiodic_approx( X, complex(y), ds, bw; basis="cheb", active_set=TestFunctionNonPeriodic.AS ) 
ANOVAapprox.approximate(a2, lambda=λs, max_iter=max_iter, precondition=false)

r = ANOVAapprox.get_AttributeRanking( a2, 0.0 )
d = ANOVAapprox.get_L2error( a2, TestFunctionNonPeriodic.norm(), TestFunctionNonPeriodic.fc ) 
println( ANOVAapprox.get_l2error(a2) )
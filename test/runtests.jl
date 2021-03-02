using ANOVAapprox
using Distributions
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

a = ANOVAapprox.periodic_approx( X, complex(y), ds, bw; method = "lsqr" ) 

ANOVAapprox.approximate(a, lambda=λs, max_iter=max_iter)

bw = [ 128, 32 ]
a2 = ANOVAapprox.periodic_approx( X, complex(y), ds, bw; method = "lsqr", active_set=TestFunction.AS ) 

ANOVAapprox.approximate(a2, lambda=λs, max_iter=max_iter)

d = ANOVAapprox.get_L2error( a2, TestFunction.norm(), TestFunction.fc ) 

@test d[0.0] < 5*10^(-3)

# Friedman 1
function f1( x::Vector{Float64} )::Float64
    if ( minimum(x) < 0 ) || ( maximum(x) > 1 )
        error( "The nodes need to be between zero and one." )
    end

    return 10*sin(pi*x[1]*x[2])+20*((x[3]-0.5)^2)+10*x[4]+5*x[5]
end

f1_active_set = Vector{Vector{Int64}}(undef, 7)
f1_active_set[1] = []
f1_active_set[2] = [1,]
f1_active_set[3] = [2,]
f1_active_set[4] = [3,]
f1_active_set[5] = [4,]
f1_active_set[6] = [5,]
f1_active_set[7] = [1,2]

Random.seed!(12334)

d = 10
ds = 2

M_train = 200
M_test = 1000

X_train = rand( d, M_train )
X_test = rand( d, M_test )

dist = Normal( 0.0, 1.0 )
noise_train = rand( dist, M_train )
noise_test = rand( dist, M_test )

y_train = [ f1(X_train[:,i])+noise_train[i] for i = 1:M_train ]
y_test = [ f1(X_test[:,i])+noise_test[i] for i = 1:M_test ]

N = Dict( 1 => [8,4], 2 => [10, 4] )
mses = ANOVAapprox.testBandwidths( X_train, complex(y_train), X_test, complex(y_test), ds, N; active_set=f1_active_set, verbose=true )
min_mses = findmin(mses)

@test min_mses[1] < 1.3
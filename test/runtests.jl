using ANOVAapprox

# initialization ##################################################

rng = MersenneTwister(1234)

d = 5
ds = 3
M = 10_000
sigma = 0.1
max_iter = 50
bw = [ 2^8, 2^5, 2^3 ]
λs = exp.(range(-2, 7, length = 30))
sort!( λs, rev=true )

X = rand( rng, d, M ) .- 0.5
y = [ TestFunction.f(X[:,i]) for i = 1:M ]

noise = zeros( Float64, M )
if sigma > 0.0
    noise = sigma*(maximum(y)-minimum(y))*randn(rng, Float64, M)
end

f = ANOVAapprox.periodic_approx( X, complex(y+noise), ds, bw; method = "lsqr" ) 

# print setting ##################################################

println("==== Approximation Test ====")
println("d = ", d, ", ds = ", ds)
println("M = ", M)
println("sigma = ", sigma)
println("max_iter = ", max_iter)
println("N = ", bw)
println("|Ids| = ", ANOVAapprox.get_NumFreq(f))
println( "oversampling = ", M/ANOVAapprox.get_NumFreq(f) )

# computations ##################################################

ANOVAapprox.approximate(f, lambda=λs, max_iter=max_iter)

# plotting ##################################################

ANOVAapprox.plot(f, :L2error, TestFunction.norm(), TestFunction.fc, :gsi, TestFunction.AS, :l2error) |> display
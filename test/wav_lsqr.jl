#### PERIODIC TEST SOLVER LSQR ####
using ANOVAapprox
include("TestFunctionPeriodic.jl")
using Test
using Random
using Aqua

d = 6
ds = 2
M = 10_000
max_iter = 50
bw = [4, 4]
λs = [0.0, 1.0]


X = rand(d, M) .- 0.5
y = [TestFunctionPeriodic.f(X[:, i]) for i = 1:M]
X_test = rand(d, M) .- 0.5
y_test = [TestFunctionPeriodic.f(X_test[:, i]) for i = 1:M]

####  ####

ads = ANOVAapprox.approx(X, y, ds, bw, "wav2")
ANOVAapprox.approximate(ads, lambda = λs)

println("AR: ", sum(ANOVAapprox.get_AttributeRanking(ads, 0.0)))
@test abs(sum(ANOVAapprox.get_AttributeRanking(ads, 0.0)) - 1) < 0.0001

bw = ANOVAapprox.get_orderDependentBW(TestFunctionPeriodic.AS, [4, 4])

aU = ANOVAapprox.approx(X, y, TestFunctionPeriodic.AS, bw, "wav2")
ANOVAapprox.approximate(aU, lambda = λs)

err_l2_ds = ANOVAapprox.get_l2error(ads)[0.0]
err_l2_U = ANOVAapprox.get_l2error(aU)[0.0]
err_l2_rand_ds = ANOVAapprox.get_l2error(ads, X_test, y_test)[0.0]
err_l2_rand_U = ANOVAapprox.get_l2error(aU, X_test, y_test)[0.0]

println("== PERIODIC LSQR ==")
println("l2 ds: ", err_l2_ds)
println("l2 U: ", err_l2_U)
println("l2 rand ds: ", err_l2_rand_ds)
println("l2 rand U: ", err_l2_rand_U)

@test err_l2_ds < 0.01
@test err_l2_U < 0.005
@test err_l2_rand_ds < 0.01
@test err_l2_rand_U < 0.005

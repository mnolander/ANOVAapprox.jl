module ANOVAapprox

using GroupedTransforms
using StatsBase
using LinearAlgebra, IterativeSolvers, LinearMaps, Plots, Distributed

abstract type fun_approx end

function get_l2error( approx::fun_approx, lambda::Float64 )::Float64
    return norm(approx.y - approx.trafo*approx.fc[lambda])/norm(approx.y)
end

function get_l2error( approx::fun_approx, X::Matrix{Float64}, y::Vector{ComplexF64} )
    ys = evaluate( approx, X )
    return Dict( λ => norm(ys[λ]-y)/norm(y) for λ in collect(keys(approx.fc)))
end

function get_l2error( approx::fun_approx )
    return Dict( λ => get_l2error(approx, λ) for λ in collect(keys(approx.fc)))
end

function get_GSI( approx::fun_approx, lambda::Float64; dict=false )
    norms = GroupedTransforms.norms( approx.fc[lambda] ).^2
    norms[1] = 0.0
    var = sum( norms )
    gsi = norms ./ var

    if dict == true 
        gsis_d = Dict()
        for i = 1:length(approx.U)
            u = approx.U[i]
            gsis_d[u] = gsi[i]
        end
    end

    return ( (dict == false) ? gsi : gsis_d )
end

function get_GSI( approx::fun_approx; dict=false )
    return Dict( λ => get_GSI(approx, λ, dict=dict) for λ in collect(keys(approx.fc)) )
end

function get_NumFreq( approx::fun_approx )::Int 
    return sum( s -> prod(s[:bandwidths].-1), approx.trafo.setting )
end

function sobolev_weights( setting; smoothness::Float64 = 2.0 )::GroupedCoeff
    what = GroupedCoeff(setting)
    for s in setting
        if s[:u] == []
            what[[]] = [1.0+0.0im]
            continue
        end
        index_set = NFFTtools.nfft_index_set_without_zeros(s[:bandwidths])
        if ndims(index_set) == 1
            what[s[:u]] = Complex.([ (1 .+ abs.(float(k)))^smoothness for k in index_set ])
        else
            what[s[:u]] = Complex.([ prod(1 .+ abs.(float(k)))^smoothness for k in eachcol(index_set) ])
        end
    end
    return what
end

function get_ActiveSet( approx::fun_approx, eps::Vector{Float64} )
    as = Dict()
    gsi = get_GSI( approx )

    for (lambda, gsi) in gsi 
        as_lambda = Vector{Vector{Int64}}(undef, length(approx.U))
        j = 1
    
        for i=1:length(approx.U)
            u = approx.U[i]
            if u == [] || gsi[i] > eps[length(u)]
                as_lambda[j] = u
                j += 1
            end
        end

        as[lambda] = as_lambda[1:j-1]
    end
    return as
end

function get_AttributeRanking( approx::fun_approx )
    return Dict( λ => get_AttributeRanking(approx, λ) for λ in collect(keys(approx.fc)) )
end

function get_AttributeRanking( approx::fun_approx, lambda::Float64 )::Vector{Float64}
    gsi = get_GSI(approx)[lambda]
    d = size(approx.X, 1)
    r = zeros(d)
    nf = 0.0

    for i = 1:length(approx.U)
        u = approx.U[i]
        if u == []
            continue
        end
        nf += gsi[i] * 1/(binomial(d,length(u)))
    end

    for i = 1:length(approx.U)
        u = approx.U[i]
        if u == []
            continue
        end
        for s in u
            r[s] += 1/(binomial(d-1,length(u)-1)) * gsi[i]
        end
    end

    r ./= d*nf

    return r
end

function get_MSE( approx::fun_approx, X::Matrix{Float64}, y::Vector{ComplexF64}; data_trafo=false )
    return Dict( λ => get_MSE(approx, X, y, λ, data_trafo=data_trafo) for λ in collect(keys(approx.fc)) )
end

function get_MSE( approx::fun_approx, X::Matrix{Float64}, y::Vector{ComplexF64}, lambda::Float64; data_trafo=false )::Float64
    y_test_approx = evaluate( approx, X )[lambda]

    if data_trafo != false 
        y_test_approx = StatsBase.reconstruct( data_trafo, real(y_test_approx) )
    end

    mse = 0.0
    
    for i = 1:length(y_test_approx)
        mse += abs( y_test_approx[i] - y[i] )^2
    end

    return mse/length(y_test_approx)
end

include( "fista.jl" )
include( "periodic_approx.jl" )
include( "periodic_approx_scat_lsqr.jl" )
include( "periodic_approx_scat_fista.jl" )
include( "nperiodic_approx.jl" )
include( "nperiodic_approx_scat_lsqr.jl" )
include( "nperiodic_approx_scat_fista.jl" )
include( "plotting.jl" )

end # module

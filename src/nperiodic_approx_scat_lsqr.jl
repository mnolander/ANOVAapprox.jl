mutable struct nperiodic_approx_scat_lsqr{d,ds} <: nperiodic_approx 
    basis::String
    X::Matrix{Float64}
    y::Vector{ComplexF64}
    U::Vector{Vector{Int64}}
    trafo::GroupedTransform
    N::Vector{Int64}
    fc::Dict{Float64,GroupedCoeff}

    function nperiodic_approx_scat_lsqr{d,ds}( basis::String, X::Matrix{Float64}, y::Vector{ComplexF64}, U::Vector{Vector{Int64}}, trafo::GroupedTransform, N::Vector{Int64} ) where {d,ds}
        return new( basis, X, y, U, trafo, N, Dict{Float64,GroupedCoeff}() )
    end 
end

function approximate( approx::nperiodic_approx_scat_lsqr{d,ds}; max_iter::Int64=30, lambda::Vector{Float64}=[0.0,], smoothness::Float64=1.5, density::Function=identity )::Nothing where {d,ds}

    what = sobolev_weights( approx.trafo.setting, smoothness=smoothness )
    M = size(approx.X,2)
    nf = get_NumFreq( approx )
    dsqrt = [ sqrt(density(X[:,i])) for i in 1:M ]

    for i = 1:length(lambda) 
        println( i, ". Lambda: ", lambda[i] )
        wsqrt = sqrt(lambda[i]).*(sqrt.(vec(what)))

        F_vec = LinearMap{ComplexF64}(
            fhat -> vcat( dsqrt.*(approx.trafo*GroupedCoeff(approx.trafo.setting, fhat)), wsqrt .* fhat ),
            f -> vec(approx.trafo'*(dsqrt.*f[1:M]))+wsqrt.*f[M+1:end],
            size(approx.X, 2)+nf, nf )

        tmp = zeros( ComplexF64, nf )
        lsqr!( tmp, F_vec, vcat(approx.y,zeros(ComplexF64,nf)), maxiter = max_iter, verbose=true )    
        approx.fc[lambda[i]] = GroupedCoeff(approx.trafo.setting, tmp)
    end

    return 
end
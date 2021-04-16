mutable struct nperiodic_approx_scat_lsqr{d,ds} <: nperiodic_approx
    basis::String
    X::Matrix{Float64}
    y::Vector{ComplexF64}
    U::Vector{Vector{Int64}}
    trafo::GroupedTransform
    N::Union{Vector{Int64},Vector{Vector{Int64}}}
    fc::Dict{Float64,GroupedCoeff}

    function nperiodic_approx_scat_lsqr{d,ds}( basis::String, X::Matrix{Float64}, y::Vector{ComplexF64}, U::Vector{Vector{Int64}}, trafo::GroupedTransform, N::Union{Vector{Int64},Vector{Vector{Int64}}} ) where {d,ds}
        return new( basis, X, y, U, trafo, N, Dict{Float64,GroupedCoeff}() )
    end
end

function cheb_density( x::Vector{Float64} )::Float64
   return prod( xs -> 1/sqrt(1-xs^2), x )
end

function getScalingVector( approx::nperiodic_approx_scat_lsqr{d,ds} )::Vector{Float64} where {d,ds}
    scalingVector = ones( Float64, get_NumFreq( approx ) )
    index = 2

    for (idx, s) in enumerate(approx.trafo.setting)
        if s[:u] == []
            continue 
        end 

        datalength = GroupedTransforms.NFCTtools.datalength( s[:bandwidths] )
        scalingVector[index:index-1+datalength] .*= sqrt(2)^length(s[:u])
        index += datalength
    end

    return scalingVector
end

function approximate( approx::nperiodic_approx_scat_lsqr{d,ds}; max_iter::Int64=30, lambda::Vector{Float64}=[0.0,], smoothness::Float64=0.0, precondition::Bool=true, verbose::Bool=false ) where {d,ds}
    if smoothness != 0.0 
        return approximate_old( approx; max_iter=max_iter, lambda=lambda, smoothness=smoothness, precondition=precondition, verbose=verbose )
    end

    M = size(approx.X, 2)
    n = get_NumFreq( approx )
    scalingVector = getScalingVector( approx )

    if ( approx.basis == "cheb" ) && precondition
        W = [ sqrt(cheb_density(approx.X[:,i])) for i in 1:M ]
    else 
        W = ones( M )
    end

    for L in lambda 
        if verbose
            println( "lambda = ", L )
        end

        F = LinearMap{ComplexF64}(
            fhat -> W.*(approx.trafo*GroupedCoeff(approx.trafo.setting, scalingVector.*fhat)),
            f -> scalingVector.*vec(approx.trafo'*(W.*f)),
            M, n )

        tmp = zeros( ComplexF64, n )
        lsqr!( tmp, F, W .* approx.y, maxiter = max_iter, verbose=verbose, damp=sqrt(L) )
        approx.fc[L] = GroupedCoeff(approx.trafo.setting, tmp)
    end

    return
end

function approximate_old( approx::nperiodic_approx_scat_lsqr{d,ds}; max_iter::Int64=30, lambda::Vector{Float64}=[0.0,], smoothness::Float64=0.0, precondition::Bool=true, verbose::Bool=false )::Nothing where {d,ds}

    what = sobolev_weights( approx.trafo.setting, smoothness=smoothness )
    M = size(approx.X,2)
    nf = get_NumFreq( approx )

    if ( approx.basis == "cheb" ) && precondition
        dsqrt = [ sqrt(cheb_density(approx.X[:,i])) for i in 1:M ]
    else 
        dsqrt = ones( M )
    end
    scalingVector = getScalingVector( approx )
    for i = 1:length(lambda)
        if verbose
            println( "lambda = ", lambda[i] )
        end
        wsqrt = sqrt(lambda[i]).*(sqrt.(vec(what)))

        F_vec = LinearMap{ComplexF64}(
            fhat -> vcat( dsqrt.*(approx.trafo*GroupedCoeff(approx.trafo.setting, scalingVector.*fhat)), wsqrt .* (scalingVector.*fhat) ),
            f -> scalingVector.*(approx, vec(approx.trafo'*(dsqrt.*f[1:M])))+wsqrt.*f[M+1:end],
            size(approx.X, 2)+nf, nf )

        tmp = zeros( ComplexF64, nf )
        lsqr!( tmp, F_vec, vcat(dsqrt .* approx.y, zeros(ComplexF64,nf)), maxiter = max_iter, verbose=verbose )
        approx.fc[lambda[i]] = GroupedCoeff(approx.trafo.setting, tmp)
    end

    return

end
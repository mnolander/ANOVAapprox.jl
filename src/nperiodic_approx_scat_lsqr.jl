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

function scaleCoefficients( approx::nperiodic_approx_scat_lsqr{d,ds}, c::Vector{ComplexF64} )::GroupedCoeff where {d,ds}
    cp = copy(c)
    fc = GroupedCoeff(approx.trafo.setting, cp)

    for u in approx.U 
        if u != []
            fc[u] = fc[u] .* (sqrt(2)^length(u))
        end
    end

    return fc
end

function scaleCoeffs( approx::nperiodic_approx_scat_lsqr{d,ds}, c::Vector{ComplexF64}; res::Integer=0 ) where {d,ds}
    fc = scaleCoefficients( approx, c )

    if res == 0 
        return fc 
    else 
        return fc.data 
    end
end

function approximate( approx::nperiodic_approx_scat_lsqr{d,ds}; max_iter::Int64=30, lambda::Vector{Float64}=[0.0,], smoothness::Float64=0.0, precondition::Bool=true, verbose::Bool=false )::Nothing where {d,ds}

    what = sobolev_weights( approx.trafo.setting, smoothness=smoothness )
    M = size(approx.X,2)
    nf = get_NumFreq( approx )

    if ( approx.basis == "cheb" ) && precondition
        dsqrt = [ sqrt(cheb_density(approx.X[:,i])) for i in 1:M ]
    else 
        dsqrt = ones( M )
    end

    for i = 1:length(lambda)
        if verbose
            println( "lambda = ", lambda[i] )
        end
        wsqrt = sqrt(lambda[i]).*(sqrt.(vec(what)))

        F_vec = LinearMap{ComplexF64}(
            fhat -> vcat( dsqrt.*(approx.trafo*scaleCoeffs( approx, fhat )), wsqrt .* scaleCoeffs( approx, fhat; res=1 ) ),
            f -> scaleCoeffs(approx, vec(approx.trafo'*(dsqrt.*f[1:M])), res=1)+wsqrt.*f[M+1:end],
            size(approx.X, 2)+nf, nf )

        tmp = zeros( ComplexF64, nf )
        lsqr!( tmp, F_vec, vcat(dsqrt .* approx.y, zeros(ComplexF64,nf)), maxiter = max_iter, verbose=verbose )
        approx.fc[lambda[i]] = GroupedCoeff(approx.trafo.setting, tmp)
    end

    return

end

abstract type nperiodic_approx <: fun_approx end

bw_vec = Union{Vector{Int64},Vector{Vector{Int64}}}

function nperiodic_approx( X_up::Matrix{Float64}, y::Vector{ComplexF64}, ds::Integer, N::bw_vec; method::String="lsqr", basis::String="cosine", active_set=false )::nperiodic_approx
    X = copy(X_up)
    d = size(X, 1)
    M = size(X, 2)

    if length(y) != M 
        error( "Length mismatch in y." )
    end

    if ds < 1 || ds > d 
        error( "Superposition dimension mismatch." )
    end

    if active_set == false
        U = get_superposition_set(d, ds)
    else
        U = active_set
    end

    if maximum(X) > 1.0 || minimum(X) < 0.0
        error( "Your nodes have to be between zero and one.")
    end

    if basis == "cosine"
        X ./= 2.0
    elseif basis == "cheb"
        X .*= 2.0
        X .-= 1.0
        X = acos.( X )
        X ./= 2.0*pi
    else 
        error("basis not implemented yet")
    end

    if isa( N, Vector{Vector{Int64}} )
        if length(U) == length(N) 
            bandwidths = N
            for i = 1:length(U)
                if length(U[i]) != length(N[i])
                    error( "bw single length mismatch" )
                end
            end
        else 
            error( "bw length mismatch" )
        end
    else
        if length(N) != ds 
            error( "bw length mismatch" )
        end
        tmp = zeros( Int64, ds+1 )
        tmp[2:end] = N
        bandwidths = [ fill(tmp[length(u)+1], length(u)) for u in U ]
    end
 
    setting = [ (u = U[idx], mode = NFCTtools, bandwidths = bandwidths[idx]) for idx in 1:length(U) ]
    F = GroupedTransform(setting, X)

    if method == "lsqr"
      return nperiodic_approx_scat_lsqr{d,ds}(basis, X, y, U, F, N)
    elseif method == "fista"
      return nperiodic_approx_scat_fista{d,ds}(basis, X, y, U, F, N)
    else
      error("method not implemented yet")
    end
end

function getIndexSet( approx::nperiodic_approx )
    I = zeros( Int64, size(approx.X,1), 1 )

    for (idx, s) in enumerate(approx.trafo.setting)
        if s[:u] == []
            continue 
        end 
        I_hat = GroupedTransforms.NFCTtools.nfct_index_set_without_zeros(s[:bandwidths])
        I_hat_2 = zeros( Int64, size(approx.X,1), ( length(s[:u]) == 1 ) ? length(I_hat) : size(I_hat,2) )
        I_hat_2[s[:u], :] = I_hat
        I = hcat( I, I_hat_2 )
    end

    return I
end

function get_L2error( approx::nperiodic_approx, norm::Float64, fc_fun::Function, lambda::Float64 ) 
    I = getIndexSet( approx )
    fc = approx.fc[lambda].data

    L2_error = norm^2

    for i = 1:length(fc)
        k = I[:,i]
        fck = fc_fun(k)
        L2_error += abs(fck - fc[i])^2 - abs( fck )^2
    end

    return sqrt(L2_error)/norm
end

function get_L2error( approx::nperiodic_approx, norm::Float64, fc_fun::Function ) 
    return Dict( λ => get_L2error(approx, norm, fc_fun, λ) for λ in collect(keys(approx.fc)))
end

function evaluate( approx::nperiodic_approx, lambda::Float64 )::Vector{ComplexF64}
    scalingVector = getScalingVector( approx )
    return approx.trafo*GroupedCoeff(approx.trafo.setting, scalingVector.*vec(approx.fc[lambda]))  
end


function evaluate( approx::nperiodic_approx, X_up::Matrix{Float64}, lambda::Float64 )::Vector{ComplexF64}
    X = copy(X_up)

    if size(X,1) != size(approx.X,1)
        error( "Matrix size mismatch." )
    end

    if approx.basis == "cosine"
        X ./= 2.0
    elseif approx.basis == "cheb"
        X .*= 2.0
        X .-= 1.0
        X = acos.( X )
        X ./= 2.0*pi
    else 
        error("basis not implemented yet")
    end

    F = GroupedTransform(approx.trafo.setting, X)
    scalingVector = getScalingVector( approx )

    return F*GroupedCoeff(approx.trafo.setting, scalingVector.*approx.fc[lambda].data)  
end

function evaluate( approx::nperiodic_approx, X::Matrix{Float64} )
    return Dict( λ => evaluate(approx, X, λ) for λ in collect(keys(approx.fc)))
end

function evaluate( approx::nperiodic_approx )
    return Dict( λ => evaluate(approx, λ) for λ in collect(keys(approx.fc)))
end

function testBandwidths( X_train::Matrix{Float64}, y_train::Vector{ComplexF64}, X_test::Matrix{Float64}, y_test::Vector{ComplexF64}, ds::Integer, N; method::String="lsqr", basis::String="cosine", active_set=false, smoothness::Float64=0.0, max_iter::Int64=1000, lambda::Vector{Float64}=[0.0,], verbose::Bool=false, data_trafo=false )

    mses_bw = Dict()

    for i in collect(keys(N))

        if !isa(N[i], bw_vec)
            error( "type mismatch" )
        end

        f = nperiodic_approx( X_train, y_train, ds, N[i]; method=method, basis=basis, active_set=active_set )
        approximate(f, smoothness=smoothness, max_iter=max_iter, lambda=lambda, verbose=verbose)
        mse = get_MSE( f, X_test, y_test, data_trafo=data_trafo )
        min_mse = findmin(mse)
        mses_bw[ (N[i], min_mse[2]) ] = min_mse[1]
        
    end

    return mses_bw

end

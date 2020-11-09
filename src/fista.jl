function bisection(fun, fval, left, right, fleft, fright; max_iter = 10, tol = 1e-15, verbose = false)
  fright -= fval
  fleft -= fval

  for iter = 1:max_iter
    global middle = (left+right)/2
    fmiddle = fun(middle).-fval

    if sign(fmiddle) == sign(fleft)
      left = middle
      fleft = fmiddle
    else
      right = middle
      fright = fmiddle
    end
    verbose && println("residual for Bisection $(abs(fmiddle))")
    abs(fmiddle) < tol && break
  end
  return middle
end

function newton(fun, dfun, fval, x; max_iter = 10, tol = 1e-15, verbose = false)
  for iter = 1:max_iter
    f = fun(x)
    df = dfun(x)
    ( isnan(f) || isnan(df) || abs(df) < 1e-15 ) && break
    x += (fval-f)/df
    verbose && println("residual for Newton: $(abs(f-fval))\nf: $(f) df: $(df) fval: $(fval) x $(x)")
    abs(f-fval) < tol && break
  end
  return x
end

function λ2ξ(λ, what, y; verbose = false)
  fun = ξ -> sum(abs.(what .* (y ./ (1/ξ .+ what)).^2 ))
  dfun = ξ -> 2*sum(abs.(what .* (y ./ (1/ξ .+ what)).^2 ./ (1/ξ .+ what)))*ξ^-2

  fright = sum(abs.(what .* (y ./ (1 .+ what)).^2))
  if λ^2 < fright
    fleft = 0
    ξ = bisection(fun, λ^2, 1e-10, 1, fleft, fright; max_iter = 25, tol = 1e-10, verbose = verbose)
  else
    fleft = sum(what .* (y ./ what).^2 )
    ξ = 1/bisection(ξ -> fun(1/ξ), λ^2, 1e-10, 1, fleft, fright; max_iter = 25, tol = 1e-16, verbose = verbose)
    ξ > 100 && return ξ
  end

  # apply Newton on f(exp(x)). Small solutions can be found more accurate this way and we work our way around negative solutions.
  ξ = newton(
    x -> fun(exp(x)),
    x -> dfun(exp(x))*exp(x),
    λ^2, log(ξ); max_iter = 50, tol = 1e-16, verbose = verbose) |> exp

  if abs(fun(ξ)-λ^2) > 1
    error("λ2ξ: something went wrong minimizing. (residual: $(abs(fun(ξ)-λ^2))")
  end

  return ξ
end

function fista!(ghat::GroupedCoeff, F::GroupedTransform, y::Vector{ComplexF64}, λ::Float64, what::GroupedCoeff; L = "adaptive", max_iter::Int = 25)
  adaptive = ( L == "adaptive" )
  if adaptive
    L = 1
    η::Int = 2
  end

  U = [ s[:u] for s in ghat.setting ]

  hhat = GroupedCoeff(ghat.setting, copy(vec(ghat)))
  t = 1.0
  val = [norm((F*hhat)-y)^2/2+λ*sum(norms(hhat, what))]

  for k = 1:max_iter-1
    ghat_old = GroupedCoeff(ghat.setting, copy(vec(ghat)))
    t_old = t
      
    Fhhat = F*hhat
    fgrad = (F'*(Fhhat-y))
    while true
      # p_L(hhat)
      set_data!(ghat, vec(hhat-1/L*fgrad))
 
      mask = map( u -> (λ/L)^2 < sum(abs.(ghat[u].^2 ./ what[u])), U)
      ξs = pmap(u -> λ2ξ(λ/L, what[u], ghat[u]), U[mask])
      for u in U[.!mask]
        ghat[u] = 0*ghat[u]
      end
      for (u, ξ) in zip(U[mask], ξs)
        if ξ == Inf
          ghat[u] = 0*ghat[u]
        else
          ghat[u] = ghat[u] ./ (1 .+ ξ*what[u])
        end
      end


      if !adaptive
        append!(val, norm((Fhhat)-y)^2/2+λ*sum(norms(hhat, what)))
        break
      end

      # F
      Fvalue = norm((F*ghat)-y)^2/2+λ*sum(norms(ghat, what))

      # Q
      Q = (norm((Fhhat)-y)^2/2
        +dot(vec(ghat-hhat), vec(fgrad))
        +L/2*norm(vec(ghat-hhat))^2
        +λ*sum(norms(ghat, what))
      )
      
      if real(Fvalue) < real(Q)+1e-10 || L >= 2^32
        append!(val, Fvalue)
        break
      else
        L *= η
      end
    end
      
    # update t
    t = (1+sqrt(1+4*t^2))/2

    # update hhat
    hhat = ghat + (t_old-1)/t*(ghat-ghat_old)

    # stoping criteria
    resnorm = norm(vec(ghat_old-ghat))
    resnorm < 1e-16 && break
    abs(val[end]-val[end-1]) < 1e-16 && break
  end
end


function plot_gsi( approx::fun_approx; colorU::Vector{Vector{Int}} = [[0]] )
  λs = sort(collect(keys(approx.fc)))

  gsi = hcat([ get_GSI(approx)[λ] for λ in λs ]...)
  maxds = maximum(s -> length(s[:u]), approx.fc[λs[1]].setting)

  maxgsi = zeros(maxds)
  for ds = 1:maxds # |u|
    for (idx, u) in enumerate(approx.U)
      if length(u) == ds
        maxgsi[ds] = maximum([maxgsi[ds], maximum(gsi[idx, :])])
      end
    end
  end

  p = []
  for ds = 1:maxds # |u|
    p = [p... , plot(title = "|u| = $ds")]
    for (idx, u) in enumerate(approx.U)
      if length(u) == ds
        clr = ( in(u, colorU) ? :orange : :black )
        plot!(λs, gsi[idx, :],
          legend = false, xlabel = "λ", ylabel = "ρ", color = clr, xaxis = :log)

        #annotate the indices of the biggest gsi
        if gsi[idx, 1] > 0.5*maxgsi[ds]
          annotate!([(minimum(λs), gsi[idx, 1], text("$u", 11))])
        end
      end
    end
  end

  plot(p..., layout = (1, maxds))
end


function Plots.:plot(approx::fun_approx, args...)
  λs = sort(collect(keys(approx.fc)))

  p = []

  # Global sensitivity indices
  if sum(args .== :gsi) > 0
    idx = findfirst(args .== :gsi)
    if length(args) > idx && isa(args[idx+1], Vector{Vector{Int}})
      p = [p..., plot_gsi(approx; colorU = args[idx+1])]
    else
      p = [p..., plot_gsi(approx)]
    end
  end

  # discrete l2-error
  if sum(args .== :l2error) > 0
    idx = findfirst(args .== :l2error)
    if length(args) > idx+1 && isa(args[idx+1], Matrix) && isa(args[idx+2], Vector)
      l2error = get_l2error(approx, args[idx+1], args[idx+2])
    else
      l2error = get_l2error(approx)
    end
    p = [p..., plot(λs, [ l2error[λ] for λ in λs ],
      legend = false, xlabel = "λ", ylabel = "l2error", xaxis = :log)]
  end

  # L2-error
  if sum(args .== :L2error) > 0
    idx = findfirst(args .== :L2error)
    if length(args) > idx+1 && isa(args[idx+1], Float64) && isa(args[idx+2], Function)
      L2error = get_L2error(approx, args[idx+1], args[idx+2])
    else
      error("missing or wrong arguments for L2-error")
    end
    p = [p..., plot(λs, [ L2error[λ] for λ in λs ],
      legend = false, xlabel = "λ", ylabel = "L2error", xaxis = :log)]
  end

  notherplots = sum(args .== :l2error) + sum(args .== :L2error)
  if sum(args .== :gsi) > 0 && notherplots > 0
    if notherplots == 1
      plot(p..., layout = (2, 1))
    elseif notherplots == 2
      plot(p..., layout = @layout [a; a a])
    end
  else
    plot(p...)
  end
end

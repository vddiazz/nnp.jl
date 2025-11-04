#----- pkg

using LinearAlgebra
using JLD2
using NPZ
using ProgressMeter
using LoopVectorization

#### baryon number

function b_dens(y1::Array{Float64},y2::Array{Float64},y3::Array{Float64},U,d1U,d2U,d3U)

    #----- prepare grid
    
    l1 = length(y1); l2 = length(y2); l3 = length(y3)

    #----- main
    
    println()
    println("#--------------------------------------------------#")
    println()
    println("Baryon number")
    println()

    dens = zeros(Float64, l1,l2,l3)

    @showprogress 1 "Computing..." for k in 1:l3
        @tturbo for j in 1:l2, i in 1:l1
            dens[i,j,k] = -12*(U[i,j,k,1]^2 + U[i,j,k,2]^2 + U[i,j,k,3]^2 + U[i,j,k,4]^2)*(d1U[i,j,k,2]*d2U[i,j,k,4]*d3U[i,j,k,3]*U[i,j,k,1] - d1U[i,j,k,2]*d2U[i,j,k,3]*d3U[i,j,k,4]*U[i,j,k,1] - d1U[i,j,k,1]*d2U[i,j,k,4]*d3U[i,j,k,3]*U[i,j,k,2] + d1U[i,j,k,1]*d2U[i,j,k,3]*d3U[i,j,k,4]*U[i,j,k,2] - d1U[i,j,k,2]*d2U[i,j,k,4]*d3U[i,j,k,1]*U[i,j,k,3] + d1U[i,j,k,1]*d2U[i,j,k,4]*d3U[i,j,k,2]*U[i,j,k,3] + d1U[i,j,k,2]*d2U[i,j,k,1]*d3U[i,j,k,4]*U[i,j,k,3] - d1U[i,j,k,1]*d2U[i,j,k,2]*d3U[i,j,k,4]*U[i,j,k,3] + d1U[i,j,k,4]*(d3U[i,j,k,3]*(-(d2U[i,j,k,2]*U[i,j,k,1]) + d2U[i,j,k,1]*U[i,j,k,2]) + d2U[i,j,k,3]*(d3U[i,j,k,2]*U[i,j,k,1] - d3U[i,j,k,1]*U[i,j,k,2]) + (d2U[i,j,k,2]*d3U[i,j,k,1] - d2U[i,j,k,1]*d3U[i,j,k,2])*U[i,j,k,3]) + (d1U[i,j,k,2]*(d2U[i,j,k,3]*d3U[i,j,k,1] - d2U[i,j,k,1]*d3U[i,j,k,3]) + d1U[i,j,k,1]*(-(d2U[i,j,k,3]*d3U[i,j,k,2]) + d2U[i,j,k,2]*d3U[i,j,k,3]))*U[i,j,k,4] + d1U[i,j,k,3]*(d3U[i,j,k,4]*(d2U[i,j,k,2]*U[i,j,k,1] - d2U[i,j,k,1]*U[i,j,k,2]) + d2U[i,j,k,4]*(-(d3U[i,j,k,2]*U[i,j,k,1]) + d3U[i,j,k,1]*U[i,j,k,2]) + (-(d2U[i,j,k,2]*d3U[i,j,k,1]) + d2U[i,j,k,1]*d3U[i,j,k,2])*U[i,j,k,4]))
        end
    end

    return dens

    #----- data saving
    #=
    if output_format == "jld2"
        path = out*"bdens.jld2"
        @save path dens

    elseif output_format == "npy"
        npzwrite(out*"/$(type_of_grid)_$(l1)x$(l2)x$(l3)/y1.npy", y1)
        npzwrite(out*"/$(type_of_grid)_$(l1)x$(l2)x$(l3)/y2.npy", y2)
        npzwrite(out*"/$(type_of_grid)_$(l1)x$(l2)x$(l3)/y3.npy", y3)

    end
    
    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")
    =#
end

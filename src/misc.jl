#----- pkg

using LinearAlgebra
using JLD2
using NPZ

#### baryon number

function b_dens(grid_size,U,d1U,d2U,d3U)

    #----- prepare grid
    
    y1 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y1.npy")
    y2 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y2.npy")
    y3 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y3.npy")

    idx_list = []
    for i in 1:length(y1)
        for j in 1:length(y2)
            for k in 1:length(y3)
                push!(idx_list, [i,j,k])
            end
        end
    end

    #----- main
    
    println()
    println("#--------------------------------------------------#")
    println()
    println("Baryon number")
    println()

    dens = zeros(Float64, l1,l2,l3)

    for idx in idx_list 
        dens[idx[1],idx[2],idx[3]] = -12*(U[idx[1],idx[2],idx[3],0]^2 + U[idx[1],idx[2],idx[3],1]^2 + U[idx[1],idx[2],idx[3],2]^2 + U[idx[1],idx[2],idx[3],3]^2)*(d1U[idx[1],idx[2],idx[3],1]*d2U[idx[1],idx[2],idx[3],3]*d3U[idx[1],idx[2],idx[3],2]*U[idx[1],idx[2],idx[3],0] - d1U[idx[1],idx[2],idx[3],1]*d2U[idx[1],idx[2],idx[3],2]*d3U[idx[1],idx[2],idx[3],3]*U[idx[1],idx[2],idx[3],0] - d1U[idx[1],idx[2],idx[3],0]*d2U[idx[1],idx[2],idx[3],3]*d3U[idx[1],idx[2],idx[3],2]*U[idx[1],idx[2],idx[3],1] + d1U[idx[1],idx[2],idx[3],0]*d2U[idx[1],idx[2],idx[3],2]*d3U[idx[1],idx[2],idx[3],3]*U[idx[1],idx[2],idx[3],1] - d1U[idx[1],idx[2],idx[3],1]*d2U[idx[1],idx[2],idx[3],3]*d3U[idx[1],idx[2],idx[3],0]*U[idx[1],idx[2],idx[3],2] + d1U[idx[1],idx[2],idx[3],0]*d2U[idx[1],idx[2],idx[3],3]*d3U[idx[1],idx[2],idx[3],1]*U[idx[1],idx[2],idx[3],2] + d1U[idx[1],idx[2],idx[3],1]*d2U[idx[1],idx[2],idx[3],0]*d3U[idx[1],idx[2],idx[3],3]*U[idx[1],idx[2],idx[3],2] - d1U[idx[1],idx[2],idx[3],0]*d2U[idx[1],idx[2],idx[3],1]*d3U[idx[1],idx[2],idx[3],3]*U[idx[1],idx[2],idx[3],2] + d1U[idx[1],idx[2],idx[3],3]*(d3U[idx[1],idx[2],idx[3],2]*(-(d2U[idx[1],idx[2],idx[3],1]*U[idx[1],idx[2],idx[3],0]) + d2U[idx[1],idx[2],idx[3],0]*U[idx[1],idx[2],idx[3],1]) + d2U[idx[1],idx[2],idx[3],2]*(d3U[idx[1],idx[2],idx[3],1]*U[idx[1],idx[2],idx[3],0] - d3U[idx[1],idx[2],idx[3],0]*U[idx[1],idx[2],idx[3],1]) + (d2U[idx[1],idx[2],idx[3],1]*d3U[idx[1],idx[2],idx[3],0] - d2U[idx[1],idx[2],idx[3],0]*d3U[idx[1],idx[2],idx[3],1])*U[idx[1],idx[2],idx[3],2]) + (d1U[idx[1],idx[2],idx[3],1]*(d2U[idx[1],idx[2],idx[3],2]*d3U[idx[1],idx[2],idx[3],0] - d2U[idx[1],idx[2],idx[3],0]*d3U[idx[1],idx[2],idx[3],2]) + d1U[idx[1],idx[2],idx[3],0]*(-(d2U[idx[1],idx[2],idx[3],2]*d3U[idx[1],idx[2],idx[3],1]) + d2U[idx[1],idx[2],idx[3],1]*d3U[idx[1],idx[2],idx[3],2]))*U[idx[1],idx[2],idx[3],3] + d1U[idx[1],idx[2],idx[3],2]*(d3U[idx[1],idx[2],idx[3],3]*(d2U[idx[1],idx[2],idx[3],1]*U[idx[1],idx[2],idx[3],0] - d2U[idx[1],idx[2],idx[3],0]*U[idx[1],idx[2],idx[3],1]) + d2U[idx[1],idx[2],idx[3],3]*(-(d3U[idx[1],idx[2],idx[3],1]*U[idx[1],idx[2],idx[3],0]) + d3U[idx[1],idx[2],idx[3],0]*U[idx[1],idx[2],idx[3],1]) + (-(d2U[idx[1],idx[2],idx[3],1]*d3U[idx[1],idx[2],idx[3],0]) + d2U[idx[1],idx[2],idx[3],0]*d3U[idx[1],idx[2],idx[3],1])*U[idx[1],idx[2],idx[3],3]))
    end

    return dens

    #----- data saving

    if output_format == "jld2"
        path = out*"/$(type_of_grid)_$(l1)x$(l2)x$(l3)/grid.jld2"
        @save path y1,y2,y3

    elseif output_format == "npy"
        npzwrite(out*"/$(type_of_grid)_$(l1)x$(l2)x$(l3)/y1.npy", y1)
        npzwrite(out*"/$(type_of_grid)_$(l1)x$(l2)x$(l3)/y2.npy", y2)
        npzwrite(out*"/$(type_of_grid)_$(l1)x$(l2)x$(l3)/y3.npy", y3)

    end

    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")
end

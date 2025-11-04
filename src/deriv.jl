#----- pkg

using JLD2
using NPZ
using LinearAlgebra
using SpecialFunctions
using ProgressMeter
using LoopVectorization

#####

function deriv_y(dir::String,grid_size::String,hD::Float64,r_idx::Int,Q_idx::Int, model::String, out::String,output_format::String)

    if (output_format != "jld2") && (output_format != "npy")
       println("invalid output data type")
    end
    
    #----- params

    field = npzread("/home/velni/phd/w/nnp/data/prod/$(model)/$(grid_size)/U_sym_r=$(r_idx)_Q=$(Q_idx).npy") # KICK OUTSIDE OF FUNCTION
    @assert eltype(field) == Float64
    field :: Array{Float64}

    l1 = length(field[:,1,1,1]); l2 = length(field[1,:,1,1]); l3 = length(field[1,1,:,1])

    d_vals = zeros(Float64, l1,l2,l3,4)
    @assert eltype(field) == Float64
    d_vals :: Array{Float64}

    step = 12*hD

    #----- main loops

    if dir == "1"

        println()
        println("#--------------------------------------------------#")
        println()
        println("Derivative in y1 direction")
        println()

        @showprogress 1 "Computing..." for k in 3:l3-2
            @tturbo for j in 3:l2-2, i in 3:l1-2, c in 1:4
                p1 = field[i+1,j,k,c]
                p2 = field[i+2,j,k,c]
                m1 = field[i-1,j,k,c]
                m2 = field[i-2,j,k,c]
                        
                d_vals[i,j,k,c] = (-p2 + 8*p1 - 8*m1 + m2)/step
            end
        end

    elseif dir == "2"

        println()
        println("#--------------------------------------------------#")
        println()
        println("Derivative in y2 direction")
        println()

        @showprogress 1 "Computing..." for k in 3:l3-2
            @tturbo for j in 3:l2-2, i in 3:l1-2, c in 1:4
                p1 = field[i,j+1,k,c]
                p2 = field[i,j+2,k,c]
                m1 = field[i,j-1,k,c]
                m2 = field[i,j-2,k,c]
                        
                d_vals[i,j,k,c] = (-p2 + 8*p1 - 8*m1 + m2)/step
            end
        end
    
    elseif dir == "3"

        println()
        println("#--------------------------------------------------#")
        println()
        println("Derivative in y3 direction")
        println()
        
        @showprogress 1 "Computing..." for k in 3:l3-2
            @tturbo for j in 3:l2-2, i in 3:l1-2, c in 1:4
                p1 = field[i,j,k+1,c]
                p2 = field[i,j,k+2,c]
                m1 = field[i,j,k-1,c]
                m2 = field[i,j,k-2,c]
                        
                d_vals[i,j,k,c] = (-p2 + 8*p1 - 8*m1 + m2)/step
            end
        end
    end
 
    #----- data saving

    if output_format == "jld2"
        path = out*"/d$(dir)U_r=$(r_idx)_Q=$(Q_idx).jld2"
        @save path d_vals

    elseif output_format == "npy"
        npzwrite(out*"/d$(dir)U_r=$(r_idx)_Q=$(Q_idx).npy", d_vals)
    end

    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")

end




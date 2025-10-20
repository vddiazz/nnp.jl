#----- pkg

using LinearAlgebra
using JLD2
using NPZ

#----- r values

function make_r(out::String,output_format::String)

    if (output_format != "jld2") && (output_format != "npy")
        println("invalid output data type")
        return
    end

    #----- r values

    r_vals = collect(1.731:0.1:7.731)

    #----- data saving

    if output_format == "jld2"
        path = out*"/r_vals.jld2"
        @save path r_vals

    elseif output_format == "npy"
        npzwrite(out*"/r_vals.npy", r_vals)
    end

    println()
    println("data saved at "*out )
    println()

    return r_vals
end

#----- Q values
#=
function make_Q(out::String,output_format::String)

    if (output_format != "jld2") && (output_format != "npy")
        println("invalid output data type")
        return
    end

    #----- Q values

    vals = [-1./2, 1./2]

    pr_1 = Float64[]
    pr_2 = Float64[]
    pr_3 = Float64[]
    pr_4 = Float64[]

    uvw_comb = Float64[]
    uvw_str = Float64[]

    for u in vals
        for v in vals
            for w in vals
                ell =  1./sqrt(1 + u^2 + v^2 + w^2)

                push!(pr_1, ell .* [1,u,v,w])
                push!(uvw_comb, [1,u,v,w])
                push!(uvw_str, ["1","u","v","w"])
=#
##### y sampling

function make_grid(type_of_grid::String, l1::Int64,l2::Int64,l3::Int64,step::Float64, out::String,output_format::String)

    yt1 = collect(-l1/2:step:l1/2)
    yt2 = collect(-l2/2:step:l2/2)
    yt3 = collect(-l3/2:step:l3/2)

    #----- main
    
    println()
    println("#--------------------------------------------------#")
    println()
    println("Grid generation")
    println()

    if type_of_grid == "reg"
        y1 = yt1
        y2 = yt2
        y3 = yt3

        print("done: reg")
    #elseif
    end

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
    println("data saved at "*out*"/$(type_of_grid)" )
    println()
    println("#--------------------------------------------------#")
end

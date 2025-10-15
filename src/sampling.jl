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



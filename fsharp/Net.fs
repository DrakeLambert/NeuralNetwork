module NeuralNetwork.Net

type Node = {
    Bias    : float
    Weights : float list
}

let createNode bias weights =
    {Bias = bias; Weights = weights}

let rec computeNet f net input =
    let computeLayer (layer : Node list) =
        [
            for node in layer -> f node input
        ]
    
    match net with
    | []    -> input
    | l::ls -> computeNet f ls (computeLayer l)  

let cost f inOut net =
    [
        for (i, o) in inOut ->
            computeNet f net i
            |> List.zip o
            |> List.sumBy (fun (expected, actual) -> pown (actual - expected) 2)
    ]
    |> List.sum
    |> (*) (1.0 / (2.0 * (List.length inOut |> float)))
    
let rec dotProduct x y = 
    match x with
    | []   -> 0.0
    | x::xs ->
        match y with
        | []    -> 0.0
        | y::ys -> x * y + dotProduct xs ys             

let appendLayer (layer : Node list) net =
    List.append net [layer]

module Defaults =    

    let defaultNode weightCount = createNode 1.0 [for _ in [0..weightCount] -> 1.0]

    let defaultLayer nodeCount =
        [for _ in [1..nodeCount] -> defaultNode nodeCount]

    let defaultNet layerCount nodeCount =
        [       
            for _ in [1..layerCount] -> defaultLayer nodeCount
        ]

    let customNet layerSizes =
        [
            for size in layerSizes -> defaultLayer size
        ]
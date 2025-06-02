module NeuralNetwork.Net

open System

type T = T of (float * float list) list list

let create inputs layerSizes =

    let rnd = Random()
    let b = rnd.NextDouble()

    let rec createR i ls =
        match ls with
        | [] -> []
        | layer::layers ->
            [for _ in [1..layer] -> b, [1.0..float i]]::createR layer layers

    createR inputs layerSizes |> T

let rec eval net input =

    let reLU x = Math.Max(0, x)

    let dot x y = List.zip x y |> List.sumBy (fun (x', y') -> x' * y')

    match net with
    | []    -> input
    | l::ls -> l |> List.map (fun (b, w) -> dot w input - b |> reLU) |> eval ls

let cost desired actual =
    let diff = (List.map2 (-) desired actual |> Vector.magnitude)
    diff * diff

let costN (desired : float list list) (actual : float list list) =
    List.zip desired actual 
    |> List.sumBy (fun (d, a) -> cost d a) 
    |> (*) (
        List.length desired 
        |> float 
        |> (/) 1.0)
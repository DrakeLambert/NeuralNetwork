module NeuralNetwork.Program

    open Net

    let layerCount = 10

    let nodeCount = 10

    [<EntryPoint>]
    let main argv =

        let net = Net.create 3 [38; 20; 10]
        printfn "%A" net

        let stopWatch = System.Diagnostics.Stopwatch.StartNew()
        [100.0..100.0..300.0]
        |> eval sigmoid net
        |> printfn "%A"
        
        stopWatch.Stop()
        
        printfn "MS: %f" stopWatch.Elapsed.TotalMilliseconds 
        
        0
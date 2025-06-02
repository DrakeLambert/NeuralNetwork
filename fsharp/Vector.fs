module Vector

type T = T of float list

let magnitude = List.sumBy (fun v -> pown v 2) >> sqrt
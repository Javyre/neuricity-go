package main

import (
    "testing"
    "fmt"
    "math"
    "math/rand"
    "sync"
)


type Node struct {
    Value float64
    NetValue float64
    Weights []float64
    Bias float64
    Error float64
}

type Layer interface {
    ForwardPass(Layer)
    GetOutputs()         []float64
    GetWeights()         [][]float64
    GetErrors()          []float64
    BackPropErr(Layer, Layer, []float64)
    UpdateWeights(Layer, float64)
}

type InputLayer []float64
type OutputLayer []Node
type HiddenLayer []Node

type Network struct {
    input InputLayer
    output OutputLayer
    hidden []HiddenLayer

    layers []Layer

    learningRate float64
}

func sigma(x float64) float64 {
    return 1 / (1 + math.Exp(-x))
}

func NewNetwork(ins, outs, layerCount, neurons uint, learningRate float64) *Network {
    input  := make(InputLayer, ins)
    output := make(OutputLayer, outs)
    hidden := make([]HiddenLayer, layerCount)
    layers := make([]Layer, layerCount+2)

    // Populate global layers pointer array 
    layers[0]             = &input
    layers[len(layers)-1] = &output
    for h := range hidden {
        layers[h+1] = &hidden[h]
    }

    // Populate output weights & biases
    for o := range output {
        node := &output[o]

        node.Weights = make([]float64, neurons)
        node.Bias    = rand.Float64()

        for w := range node.Weights {
            node.Weights[w] = rand.Float64()
        }
    }

    // Populate hidden weights & biases
    for h := range hidden {
        n := neurons; if h == 0 { n = ins }
        hidden[h] = make(HiddenLayer, neurons)

        for i := range hidden[h] {
            node := &hidden[h][i]

            node.Weights = make([]float64, n)
            node.Bias    = rand.Float64()

            for w := range node.Weights {
                node.Weights[w] = rand.Float64()
            }
        }
    }

    return &Network {
        input,
        output,
        hidden,

        layers,

        learningRate,
    }
}

func (net *Network) Train(input, targetOutput []float64) { 
    net.ForwardPass(input)
    net.BackProp(targetOutput)
}

func (net *Network) ForwardPass(input []float64) {
    // Populate input values
    for i := range net.input {
        net.input[i] = input[i]
    }

    for l := range net.layers {
        if l > 0 {
            net.layers[l].ForwardPass(net.layers[l-1])
        }
    }

}

func (net *Network) BackProp(targetOutput []float64) {
    for l := range net.layers {
        l = len(net.layers)-l-1

        var next Layer = nil
        var prev Layer = nil
        if l > 0 {
            prev = net.layers[l-1] 
        }
        if l < len(net.layers)-1 {
            next = net.layers[l+1]
        }

        net.layers[l].BackPropErr(prev, next, targetOutput)
        net.layers[l].UpdateWeights(prev, net.learningRate)
    }
}

func (net *Network) GetError(targetOutput []float64) float64 {
    e := 0.
    for o := range targetOutput {
        e += math.Pow(targetOutput[o] - net.output[o].Value, 2) / 2
    }

    return e
}

func (net *Network) GetOutput() []float64 {
    return net.output.GetOutputs()
}

// ==================== ForwardPass() ====================

/// DUMMY
func (l *InputLayer) ForwardPass(prev_l Layer) {
}

func (l *OutputLayer) ForwardPass(prev_l Layer) { 
    var wg sync.WaitGroup
    wg.Add(len(*l))
    defer wg.Wait()

    prev_l_outs := prev_l.GetOutputs()

    for n := range *l {
        go func(n int) {
            defer wg.Done()

            node := &(*l)[n]

            sum := node.Bias
            for i, v := range prev_l_outs {
                sum += v * node.Weights[i]
            }

            node.NetValue = sum
            node.Value = sigma(sum)
        }(n)
    }
}

func (l *HiddenLayer) ForwardPass(prev_l Layer) {
    (*OutputLayer)(l).ForwardPass(prev_l)
}

// ==================== BackProp() ====================

/// DUMMY
func (l *InputLayer) BackPropErr(prev_l, next_l Layer, targetOutput []float64) {
}

func (l *OutputLayer) BackPropErr(prev_l, next_l Layer, targetOutput []float64) {
    var wg sync.WaitGroup
    wg.Add(len(*l))
    defer wg.Wait()
    // calculate error terms
    for o := range *l {
        go func(o int) {
            defer wg.Done()

            node := &(*l)[o]

            node.Error = node.Value * (1-node.Value) * (node.Value - targetOutput[o])
        }(o)
    }
}

func (l *HiddenLayer) BackPropErr(prev_l, next_l Layer, targetOutput []float64) {
    var wg sync.WaitGroup
    wg.Add(len(*l))
    defer wg.Wait()
    // calculate error terms
    for h := range *l {
        go func(h int) {
            defer wg.Done()

            node := &(*l)[h]

            sum := 0.
            n_weights := next_l.GetWeights()
            n_errs    := next_l.GetErrors()
            for n := range n_weights {
                sum += n_weights[n][h] * n_errs[n]
            }
            node.Error = node.Value * (1-node.Value) * sum
        }(h)
    }

}

// ==================== UpdateWeights) ====================

func (l *InputLayer) UpdateWeights(prev_l Layer, learningRate float64) {
}

func (l *OutputLayer) UpdateWeights(prev_l Layer, learningRate float64) {
    for o := range *l {
        node := &(*l)[o]

        for w := range node.Weights {
            node.Weights[w] -= prev_l.GetOutputs()[w] * learningRate * node.Error
        }
        node.Bias -= learningRate * node.Error
    }
}

func (l *HiddenLayer) UpdateWeights(prev_l Layer, learningRate float64) {
    (*OutputLayer)(l).UpdateWeights(prev_l, learningRate)
}

// ==================== GetOutputs() ====================

func (l *InputLayer) GetOutputs() []float64 {
    return *l
}

func (l *OutputLayer) GetOutputs() []float64 { 
    outs := make([]float64, len(*l))
    for o := range outs {
        outs[o] = (*l)[o].Value
    }

    return outs
}

func (l *HiddenLayer) GetOutputs() []float64 {
    return (*OutputLayer)(l).GetOutputs()
}

// ==================== GetWeights() ====================

/// DUMMY
func (l *InputLayer) GetWeights() [][]float64 {
    return [][]float64{}
}

func (l *OutputLayer) GetWeights() [][]float64 {
    outs := make([][]float64, len(*l))
    for o := range outs {
        outs[o] = (*l)[o].Weights
    }

    return outs
}

func (l *HiddenLayer) GetWeights() [][]float64 {
    return (*OutputLayer)(l).GetWeights()
}

// ==================== GetErrors() ====================

/// DUMMY
func (l *InputLayer) GetErrors() []float64 {
    return []float64{}
}

func (l *OutputLayer) GetErrors() []float64 {
    outs := make([]float64, len(*l))
    for o := range outs {
        outs[o] = (*l)[o].Error
    }

    return outs
}

func (l *HiddenLayer) GetErrors() []float64 {
    return (*OutputLayer)(l).GetErrors()
}


func generateList(list *[]float64) {
    for i := range *list {
        // r[i] = rand.Float64() + float64(rand.Int())
        (*list)[i] = float64(rand.Intn(10))/10
    }
}

func reverseList(from, dest *[]float64) {
    max := (len(*from)-1)

    for i := range *dest {
        (*dest)[i] = (*from)[max-i]
    }
}

func TestCorrob(t *testing.T) {
    is_eq := func (a, b []float64) bool {
        if len(a) != len(b) { return false }
        for i := range a {
            if a[i] != b[i] { return false }
        }
        return true
    }

    net := NewNetwork(2, 2, 1, 2, 0.5)
    net.hidden = []HiddenLayer{
        {
            Node {
                Weights: []float64{0.15, 0.20},
                Bias:    .35,
            },
            Node {
                Weights: []float64{0.25, 0.30},
                Bias:    .35,
            },
        },
    }
    net.output = OutputLayer {
        Node {
            Weights: []float64{0.40, 0.45},
            Bias:    .60,
        },
        Node {
            Weights: []float64{0.50, 0.55},
            Bias:    .60,
        },
    }

    net.layers = make([]Layer, 3)
    net.layers[0] = &net.input
    net.layers[1] = &net.hidden[0]
    net.layers[2] = &net.output

    input  := []float64 { 0.05, 0.10 }
    output := []float64 { 0.01, 0.99 }

    net.Train(input, output)
    if !is_eq(
        net.GetOutput(),
        []float64 { 0.7513650695523157, 0.7729284653214625 },
    ) { 
        t.Fail()
    }

    if !is_eq(
        net.hidden[0].GetWeights()[0],
        []float64 {0.14981763856120295, 0.19963527712240592},
    ) {
        t.Fail()
    }
    if !is_eq(
        net.output.GetWeights()[0],
        []float64 {0.35891647971788465, 0.4086661860762334},
    ) {
        t.Fail()
    }
}

func main() {
    rand.Seed(13)

    vals := uint(4)

    net := NewNetwork(vals, vals, 10, 6, 0.8)

    // input :=  []float64 { 1, 2, 3, 4 }
    // output := []float64 { 4, 3, 2, 1 }
    // input :=  []float64 { 1, 2 }
    // output := []float64 { 2, 1 }

    input  := make([]float64, vals)
    output := make([]float64, vals)
    oldError := 0.
    maxError := 0.
    ii := 0
    for t := 0; t < 100000000; t++ {
        generateList(&input)
        reverseList(&input, &output)

        for i := 0; i < 1; i++ {
            net.Train(input, output)
            newError := net.GetError(output)
            if i == 0 {
                maxError = newError
            }
            if ii % 1 == 0 && t > 100000 {
                dErr := newError-oldError
                if ii == 0 { dErr = 0 }
                fmt.Printf(
                    "%d\t%f\t%f\t%f\t%v\t%v\t%v\n",
                    ii,
                    newError,
                    dErr,
                    maxError,
                    input,
                    output,
                    net.GetOutput(),
                )
                // fmt.Println("input:     ", input)
                // fmt.Println("target:    ", output)
                // fmt.Println("output:    ", net.GetOutput())
                // fmt.Println("deltaError:", newError - oldError)
                // fmt.Println("error:     ", newError)
                // fmt.Println("-----------")
            }
            oldError = newError
        ii++
        }
    }
}

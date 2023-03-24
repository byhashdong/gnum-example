package main

import (
	"fmt"
	"log"
	"math"

	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

func main() {
	// x ** 2 - 2 * x + 5
	fcn := func(p []float64) float64 {
		return math.Pow(p[0], 2.0) - 2*p[0] + 5
	}

	grad := func(grad, x []float64) {
		fd.Gradient(grad, fcn, x, nil)
	}

	hess := func(h *mat.SymDense, x []float64) {
		fd.Hessian(h, fcn, x, nil)
	}

	p := optimize.Problem{
		Func: fcn,
		Grad: grad,
		Hess: hess,
	}

	var meth = &optimize.LBFGS{}
	var p0 = []float64{2.0, 3.0}

	res, err := optimize.Minimize(p, p0, nil, meth)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%#v\n", res.Location)
}

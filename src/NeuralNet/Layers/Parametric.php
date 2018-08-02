<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;

interface Parametric extends Layer
{
    const SCALE = 1e8;

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int;

    /**
     * Compute the training activations of each neuron in the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function forward(Matrix $input) : Matrix;

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function infer(Matrix $input) : Matrix;

    /**
     * Read the parameters and return them in an associative array.
     *
     * @return array
     */
    public function read() : array;

    /**
     * Restore the parameters in the layer.
     *
     * @param  array  $parameters
     * @return void
     */
    public function restore(array $parameters) : void;
}

<?php

namespace Rubix\Engine\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;

interface Parametric extends Layer
{
    /**
     * Generate a random weight for each synapse in the layer.
     *
     * @param  \Rubix\Engine\NeuralNet\Layers\Layer
     * @return void
     */
    public function initialize(Layer $previous) : void;

    /**
     * Update the parameters in the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $steps
     * @return void
     */
    public function update(Matrix $steps) : void;

    /**
     * Restore the parameters in the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $weights
     * @return void
     */
    public function restore(Matrix $weights) : void;

    /**
     * The width of the layer. i.e. the number of neurons or computation nodes.
     *
     * @return int
     */
    public function width() : int;

    /**
     * Return the weight matrix.
     *
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function weights() : Matrix;

    /**
     * The memoized activations of the last forward pass.
     *
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function computed() : Matrix;

    /**
     * Return an error matrix computed from last backward pass.
     *
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function errors() : Matrix;

    /**
     * A matrix of gradients computed during the last backward pass.
     *
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function gradients() : Matrix;
}

<?php

namespace Rubix\Engine\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;

interface Parametric extends Layer
{
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
     * Return the weight matrix.
     *
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function weights() : Matrix;

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

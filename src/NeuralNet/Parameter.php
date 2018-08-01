<?php

namespace Rubix\ML\NeuralNet;

use MathPHP\LinearAlgebra\Matrix;

class Parameter
{
    /**
     * The parameter matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $w;

    /**
     * @return void
     */
    public function __construct(Matrix $w)
    {
        $this->w = $w;
    }

    /**
     * Update the parameter matrix.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $step
     * @return void
     */
    public function update(Matrix $step) : void
    {
        $this->w = $this->w->add($step);
    }

    /**
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function w() : Matrix
    {
        return $this->w;
    }
}

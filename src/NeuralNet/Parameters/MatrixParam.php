<?php

namespace Rubix\ML\NeuralNet\Parameters;

use Tensor\Matrix;
use Tensor\Tensor;

/**
 * Matrix Parameter
 *
 * This wrapper enables parameters to be identified by gradient descent optimizers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MatrixParam extends Parameter
{
    /**
     * The parameter matrix.
     *
     * @var \Tensor\Matrix
     */
    protected $w;

    /**
     * @param \Tensor\Matrix $w
     */
    public function __construct(Matrix $w)
    {
        parent::__construct();

        $this->w = $w;
    }

    /**
     * Return the parameter matrix.
     *
     * @return \Tensor\Matrix
     */
    public function w() : Matrix
    {
        return $this->w;
    }

    /**
     * Update the parameter.
     *
     * @param \Tensor\Tensor<mixed> $step
     */
    public function update(Tensor $step) : void
    {
        $this->w = $this->w->subtract($step);
    }

    /**
     * Perform a deep copy of the object.
     */
    public function __clone()
    {
        $this->w = clone $this->w;
    }
}

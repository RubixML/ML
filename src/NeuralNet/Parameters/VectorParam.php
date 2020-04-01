<?php

namespace Rubix\ML\NeuralNet\Parameters;

use Tensor\Vector;
use Tensor\Tensor;

/**
 * Vector Parameter
 *
 * This wrapper enables parameters to be identified by gradient descent optimizers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class VectorParam extends Parameter
{
    /**
     * The parameter matrix.
     *
     * @var \Tensor\Vector
     */
    protected $w;

    /**
     * @param \Tensor\Vector $w
     */
    public function __construct(Vector $w)
    {
        parent::__construct();

        $this->w = $w;
    }

    /**
     * Return the parameter matrix.
     *
     * @return \Tensor\Vector
     */
    public function w() : Vector
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
        $this->w = $this->w()->subtract($step);
    }

    /**
     * Perform a deep copy of the object.
     */
    public function __clone()
    {
        $this->w = clone $this->w;
    }
}

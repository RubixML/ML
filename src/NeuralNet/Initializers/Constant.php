<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;

/**
 * Constant
 *
 * Initialize the parameter to a user specified constant value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Constant implements Initializer
{
    /**
     * The value to initialize the parameter to.
     *
     * @var float
     */
    protected $value;

    /**
     * @param float $value
     */
    public function __construct(float $value = 0.)
    {
        $this->value = $value;
    }

    /**
     * Initialize a weight matrix W in the dimensions fan in x fan out.
     *
     * @param int $fanIn
     * @param int $fanOut
     * @return \Tensor\Matrix
     */
    public function initialize(int $fanIn, int $fanOut) : Matrix
    {
        return Matrix::fill($this->value, $fanOut, $fanIn);
    }
}

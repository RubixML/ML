<?php

namespace Rubix\ML\NeuralNet;

use Rubix\Tensor\Matrix;

/**
 * Parameter
 *
 * This wrapper enables parameters to be identified by object hash and thus
 * used as cache keys by adaptive gradient descent optimizers such as Adam,
 * AdaGrad, and RMS Prop.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Parameter
{
    /**
     * The unique identifier of this parameter.
     *
     * @var string
     */
    protected $id;

    /**
     * The parameter Matrix.
     *
     * @var \Rubix\Tensor\Matrix
     */
    protected $w;

    /**
     * @param \Rubix\Tensor\Matrix $w
     */
    public function __construct(Matrix $w)
    {
        $this->id = uniqid();
        $this->w = $w;
    }

    /**
     * Return the unique identifier of the parameter.
     *
     * @return string
     */
    public function id() : string
    {
        return $this->id;
    }

    /**
     * Return the parameter matrix.
     *
     * @return \Rubix\Tensor\Matrix
     */
    public function w() : Matrix
    {
        return $this->w;
    }

    /**
     * Update the parameter.
     *
     * @param \Rubix\Tensor\Matrix $step
     */
    public function update(Matrix $step) : void
    {
        $this->w = $this->w()->subtract($step);
    }
}

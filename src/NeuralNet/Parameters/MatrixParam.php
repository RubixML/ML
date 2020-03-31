<?php

namespace Rubix\ML\NeuralNet\Parameters;

use Tensor\Matrix;
use Tensor\Tensor;

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
}

<?php

namespace Rubix\ML\NeuralNet\Parameters;

use Rubix\Tensor\Matrix;
use Rubix\Tensor\Tensor;

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
     * @var \Rubix\Tensor\Matrix
     */
    protected $w;

    /**
     * @param \Rubix\Tensor\Matrix $w
     */
    public function __construct(Matrix $w)
    {
        parent::__construct();

        $this->w = $w;
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
     * @param \Rubix\Tensor\Tensor $step
     */
    public function update(Tensor $step) : void
    {
        $this->w = $this->w()->subtract($step);
    }
}

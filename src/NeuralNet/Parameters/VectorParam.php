<?php

namespace Rubix\ML\NeuralNet\Parameters;

use Rubix\Tensor\Vector;
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
class VectorParam extends Parameter
{
    /**
     * The parameter matrix.
     *
     * @var \Rubix\Tensor\Vector
     */
    protected $w;

    /**
     * @param \Rubix\Tensor\Vector $w
     */
    public function __construct(Vector $w)
    {
        parent::__construct();

        $this->w = $w;
    }

    /**
     * Return the parameter matrix.
     *
     * @return \Rubix\Tensor\Vector
     */
    public function w() : Vector
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

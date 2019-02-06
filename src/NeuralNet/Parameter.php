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
     * The parameter Matrix.
     *
     * @var \Rubix\Tensor\Matrix
     */
    public $w;

    /**
     * @param  \Rubix\Tensor\Matrix  $w
     * @return void
     */
    public function __construct(Matrix $w)
    {
        $this->w = $w;
    }
}

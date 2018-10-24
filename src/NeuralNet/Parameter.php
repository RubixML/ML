<?php

namespace Rubix\ML\NeuralNet;

use Rubix\Tensor\Matrix;

/**
 * Parameter
 *
 * This wrapper enables parameters to be identified by object hash and thus
 * used as cache keys by gradient descent optimizers such as Adam, AdaGrad,
 * and RMS Prop.
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
    protected $w;

    /**
     * @param  \Rubix\Tensor\Matrix  $w
     * @return void
     */
    public function __construct(Matrix $w)
    {
        $this->w = $w;
    }

    /**
     * @return \Rubix\Tensor\Matrix
     */
    public function w() : Matrix
    {
        return $this->w;
    }

    /**
     * Update the parameter Matrix.
     *
     * @param  \Rubix\Tensor\Matrix  $step
     * @return void
     */
    public function update(Matrix $step) : void
    {
        $this->w = $this->w->subtract($step);
    }

    /**
     * Allow methods to be called on the parameter matrix from the wrapper.
     *
     * @param  string  $name
     * @param  array  $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        return $this->w->$name(...$arguments);
    }
}

<?php

namespace Rubix\ML\NeuralNet;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

/**
 * Parameter
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Parameter
{
    /**
     * The auto incrementing id.
     *
     * @var int
     */
    protected static int $counter = 0;

    /**
     * The unique identifier of the parameter.
     *
     * @var int
     */
    protected int $id;

    /**
     * The parameter.
     *
     * @var Tensor
     */
    protected Tensor $param;

    /**
     * The accumulated gradient of the parameter.
     *
     * @var Tensor<int|float|array>|null
     */
    protected ?Tensor $gradient = null;

    /**
     * @param Tensor $param
     */
    public function __construct(Tensor $param)
    {
        $this->id = self::$counter++;
        $this->param = $param;
    }

    /**
     * Return the unique identifier of the parameter.
     *
     * @return int
     */
    public function id() : int
    {
        return $this->id;
    }

    /**
     * Return the wrapped parameter.
     *
     * @return mixed
     */
    public function param()
    {
        return $this->param;
    }

    /**
     * Return the accumulated gradient of the parameter.
     *
     * @return Tensor<int|float|array>|null
     */
    public function gradient()
    {
        return $this->gradient;
    }

    /**
     * Does the parameter have an accumulated gradient?
     *
     * @return bool
     */
    public function hasGradient() : bool
    {
        return isset($this->gradient);
    }

    /**
     * Accumulate the gradient of the parameter.
     *
     * @param Tensor<int|float|array> $gradient
     */
    public function accumulate(Tensor $gradient) : void
    {
        $this->gradient = $this->gradient
            ? $this->gradient->add($gradient)
            : $gradient;
    }

    /**
     * Reset the accumulated gradient of the parameter.
     */
    public function resetGradient() : void
    {
        $this->gradient = null;
    }

    /**
     * Apply a step of gradient descent to the parameter.
     *
     * @param Optimizer $optimizer
     */
    public function update(Optimizer $optimizer) : void
    {
        $step = $optimizer->update($this);

        $this->param = $this->param->subtract($step);

        $this->resetGradient();
    }

    /**
     * Perform a deep copy of the object upon cloning.
     */
    public function __clone()
    {
        $this->param = clone $this->param;

        if ($this->gradient) {
            $this->gradient = clone $this->gradient;
        }
    }
}

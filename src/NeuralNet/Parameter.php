<?php

namespace Rubix\ML\NeuralNet;

use Tensor\Tensor;
use Tensor\Vector;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\Exceptions\RuntimeException;

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
    protected ?Tensor $gradient;

    /**
     * Is the parameter frozen?
     *
     * @var bool
     */
    protected bool $frozen;

    /**
     * Flatten a matrix into a vector or return a vector as-is.
     *
     * @param Tensor $tensor
     * @return Vector
     */
    protected static function vectorize(Tensor $tensor) : Vector
    {
        if ($tensor instanceof Matrix) {
            return $tensor->flatten();
        }

        if ($tensor instanceof Vector) {
            return $tensor;
        }

        throw new RuntimeException('Unable to compute the norm of the accumulated gradient.');
    }

    /**
     * @param Tensor $param
     */
    public function __construct(Tensor $param)
    {
        $this->id = self::$counter++;
        $this->param = $param;
        $this->gradient = null;
        $this->frozen = false;
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
     * Is the parameter frozen?
     *
     * @return bool
     */
    public function frozen() : bool
    {
        return $this->frozen;
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
     * Return the L2 norm of the accumulated gradient.
     *
     * @return float
     */
    public function gradientNorm() : float
    {
        if (!$this->hasGradient()) {
            throw new RuntimeException('No gradient to compute norm.');
        }

        $tensor = self::vectorize($this->gradient);

        return $tensor->l2Norm();
    }

    /**
     * Accumulate the gradient of the parameter.
     *
     * @param Tensor<int|float|array> $gradient
     */
    public function accumulateGradient(Tensor $gradient) : void
    {
        if ($this->frozen) {
            return;
        }

        $this->gradient = $this->gradient
            ? $this->gradient->add($gradient)
            : $gradient;
    }

    /**
     * Scale the accumulated gradient by a scalar.
     *
     * @param float $scale
     */
    public function scaleGradient(float $scale) : void
    {
        if (!$this->hasGradient()) {
            throw new RuntimeException('No gradient to scale.');
        }

        $this->gradient = $this->gradient->multiply($scale);
    }

    /**
     * Apply a step of gradient descent to the parameter.
     *
     * @param Optimizer $optimizer
     */
    public function update(Optimizer $optimizer) : void
    {
        if ($this->frozen) {
            return;
        }

        $step = $optimizer->update($this);

        $this->param = $this->param->subtract($step);
    }

    /**
     * Reset the accumulated gradient of the parameter.
     */
    public function resetGradient() : void
    {
        $this->gradient = null;
    }

    /**
     * Freeze the parameter, preventing it from being updated during training.
     */
    public function freeze() : void
    {
        $this->frozen = true;
    }

    /**
     * Unfreeze the parameter, allowing it to be updated during training.
     */
    public function unfreeze() : void
    {
        $this->frozen = false;
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

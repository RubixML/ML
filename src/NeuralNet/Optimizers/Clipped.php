<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Clipped
 *
 * A gradient clipping wrapper that limits the magnitude of each gradient to a
 * given maximum absolute value before delegating the update to the wrapped
 * optimizer. Clipping the gradients prevents exploding gradients and keeps the
 * updates within a bounded range.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Clipped implements Optimizer
{
    /**
     * The optimizer whose updates this wrapper clips.
     *
     * @var Optimizer
     */
    protected Optimizer $optimizer;

    /**
     * The maximum absolute value of each element of the gradient.
     *
     * @var float
     */
    protected float $max;

    /**
     * @param Optimizer $optimizer
     * @param float $max
     * @throws InvalidArgumentException
     */
    public function __construct(Optimizer $optimizer, float $max = 1.0)
    {
        if ($max <= 0.0) {
            throw new InvalidArgumentException('Max must be'
                . " greater than 0, $max given.");
        }

        $this->optimizer = $optimizer;
        $this->max = $max;
    }

    /**
     * The underlying learning rate scheduler instance.
     *
     * @internal
     */
    public function scheduler() : Scheduler
    {
        return $this->optimizer->scheduler();
    }

    /**
     * Warm the parameter cache.
     *
     * @internal
     *
     * @param Parameter $param
     */
    public function warm(Parameter $param) : void
    {
        $this->optimizer->warm($param);
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @internal
     *
     * @param Parameter $param
     * @param Tensor<int|float|array> $gradient
     * @return Tensor<int|float|array>
     */
    public function update(Parameter $param, Tensor $gradient) : Tensor
    {
        $gradient = $gradient->clip(-$this->max, $this->max);

        return $this->optimizer->update($param, $gradient);
    }

    /**
     * Take a step of gradient descent for a set of parameters.
     *
     * @internal
     *
     * @param list<array{Parameter, Tensor<int|float|array>}> $gradients
     */
    public function step(array $gradients) : void
    {
        foreach ($gradients as [$param, $gradient]) {
            $param->update($gradient, $this);
        }

        $this->optimizer->scheduler()->tick();
    }

    /**
     * Flush the parameter cache.
     *
     * @internal
     */
    public function flush() : void
    {
        $this->optimizer->flush();
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Clipped (optimizer: {$this->optimizer}, max: {$this->max})";
    }
}

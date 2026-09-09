<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function sqrt;

/**
 * NormClipped
 *
 * A gradient clipping wrapper that limits the L2 norm of each individual
 * gradient to a given maximum before delegating the update to the wrapped
 * optimizer. When the norm of a gradient exceeds the maximum, the gradient is
 * scaled down proportionally rather than truncated element-wise.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NormClipped implements Optimizer
{
    /**
     * The optimizer whose updates this wrapper clips.
     *
     * @var Optimizer
     */
    protected Optimizer $optimizer;

    /**
     * The maximum L2 norm of a gradient.
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
        $sum = $gradient->square()->sum();

        if ($sum instanceof Tensor) {
            $sum = $sum->sum();
        }

        $norm = sqrt($sum);

        if ($norm > $this->max) {
            $gradient = $gradient->multiply($this->max / $norm);
        }

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
            $param->update($this->update($param, $gradient));
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
        return "Norm Clipped (optimizer: {$this->optimizer}, max: {$this->max})";
    }
}

<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Rubix\ML\Exceptions\RuntimeException;

use function get_class;

use const Rubix\ML\EPSILON;

/**
 * AdaGrad
 *
 * Short for Adaptive Gradient, the AdaGrad Optimizer speeds up the learning of
 * parameters that do not change often and slows down the learning of parameters
 * that do enjoy heavy activity.
 *
 * References:
 * [1] J. Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning
 * and Stochastic Optimization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AdaGrad implements Optimizer
{
    /**
     * The learning rate schedule.
     *
     * @var Scheduler
     */
    protected Scheduler $scheduler;

    /**
     * The cache of sum of squared gradients.
     *
     * @var Tensor[]
     */
    protected array $cache = [
        //
    ];

    /**
     * @param Scheduler $scheduler
     */
    public function __construct(Scheduler $scheduler)
    {
        $this->scheduler = $scheduler;
    }

    /**
     * The underlying learning rate scheduler instance.
     *
     * @internal
     */
    public function scheduler() : Scheduler
    {
        return $this->scheduler;
    }

    /**
     * Warm the parameter cache.
     *
     * @internal
     *
     * @param Parameter $param
     * @throws RuntimeException
     */
    public function warm(Parameter $param) : void
    {
        $class = get_class($param->param());

        if ($class === false) {
            throw new RuntimeException('Could not locate parameter class.');
        }

        $this->cache[$param->id()] = $class::zeros(...$param->param()->shape());
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
        $norm = $this->cache[$param->id()];

        $norm = $norm->add($gradient->square());

        $this->cache[$param->id()] = $norm;

        return $gradient->multiply($this->scheduler->rate())
            ->divide($norm->sqrt()->clipLower(EPSILON));
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

        $this->scheduler->tick();
    }

    /**
     * Flush the parameter cache.
     *
     * @internal
     */
    public function flush() : void
    {
        $this->cache = [];
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
        return "AdaGrad (scheduler: {$this->scheduler})";
    }
}

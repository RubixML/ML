<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\Helpers\Params;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function get_class;

/**
 * Momentum
 *
 * Momentum adds velocity to each step until exhausted. It does so by accumulating momentum from past updates and adding
 * a factor of the previous velocity to the current step.
 *
 * References:
 * [1] D. E. Rumelhart et al. (1988). Learning representations by back-propagating errors.
 * [2] I. Sutskever et al. (2013). On the importance of initialization and momentum in deep learning.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Momentum implements Optimizer
{
    /**
     * The learning rate schedule.
     *
     * @var Scheduler
     */
    protected Scheduler $scheduler;

    /**
     * The rate at which the momentum force decays.
     *
     * @var float
     */
    protected float $decay;

    /**
     * Should we employ Nesterov's lookahead (NAG) when updating the parameters?
     *
     * @var bool
     */
    protected bool $lookahead;

    /**
     * The parameter cache of velocity matrices.
     *
     * @var Tensor[]
     */
    protected array $cache = [
        //
    ];

    /**
     * @param Scheduler $scheduler
     * @param float $decay
     * @param bool $lookahead
     * @throws InvalidArgumentException
     */
    public function __construct(Scheduler $scheduler, float $decay = 0.1, bool $lookahead = false)
    {
        if ($decay <= 0.0 or $decay >= 1.0) {
            throw new InvalidArgumentException('Decay must be between'
                . " 0 and 1, $decay given.");
        }

        $this->scheduler = $scheduler;
        $this->decay = $decay;
        $this->lookahead = $lookahead;
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
     * Warm the cache.
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
     * @return Tensor<int|float|array>
     */
    public function update(Parameter $param) : Tensor
    {
        if (!$param->hasGradient()) {
            throw new RuntimeException('Cannot update parameter with no gradient.');
        }

        $velocity = $this->cache[$param->id()];

        $velocity = $param->gradient()->multiply($this->scheduler->rate())
            ->add($velocity->multiply(1.0 - $this->decay));

        $this->cache[$param->id()] = $velocity;

        if ($this->lookahead) {
            $velocity = $param->gradient()->multiply($this->scheduler->rate())
                ->add($velocity->multiply(1.0 - $this->decay));
        }

        return $velocity;
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
        return "Momentum (scheduler: {$this->scheduler}, decay: {$this->decay},"
            . ' lookahead: ' . Params::toString($this->lookahead) . ')';
    }
}

<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function get_class;

use const Rubix\ML\EPSILON;

/**
 * Adam
 *
 * Short for *Adaptive Moment Estimation*, the Adam Optimizer combines both
 * Momentum and RMS prop to achieve a balance of velocity and stability. In
 * addition to storing an exponentially decaying average of past squared
 * gradients like RMSprop, Adam also keeps an exponentially decaying average
 * of past gradients, similar to Momentum. Whereas Momentum can be seen as a
 * ball running down a slope, Adam behaves like a heavy ball with friction.
 *
 * References:
 * [1] D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Adam implements Optimizer
{
    /**
     * The learning rate schedule.
     *
     * @var Scheduler
     */
    protected Scheduler $scheduler;

    /**
     * The momentum decay rate.
     *
     * @var float
     */
    protected float $momentumDecay;

    /**
     * The decay rate of the previous norms.
     *
     * @var float
     */
    protected float $normDecay;

    /**
     * The parameter cache of running velocity and squared gradients.
     *
     * @var array<Tensor[]>
     */
    protected array $cache = [
        //
    ];

    /**
     * @param Scheduler $scheduler
     * @param float $momentumDecay
     * @param float $normDecay
     * @throws InvalidArgumentException
     */
    public function __construct(Scheduler $scheduler, float $momentumDecay = 0.1, float $normDecay = 0.001)
    {
        if ($momentumDecay <= 0.0 or $momentumDecay >= 1.0) {
            throw new InvalidArgumentException('Momentum decay must be'
                . " between 0 and 1, $momentumDecay given.");
        }

        if ($normDecay <= 0.0 or $normDecay >= 1.0) {
            throw new InvalidArgumentException('Norm decay must be'
                . " between 0 and 1, $normDecay given.");
        }

        $this->scheduler = $scheduler;
        $this->momentumDecay = $momentumDecay;
        $this->normDecay = $normDecay;
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

        $zeros = $class::zeros(...$param->param()->shape());

        $this->cache[$param->id()] = [clone $zeros, $zeros];
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @internal
     *
     * @param Parameter $param
     * @param Tensor<int|float|array> $gradient
     * @return Tensor<int|float|array>
     */
    public function update(Parameter $param, Tensor $gradient) : Tensor
    {
        [$velocity, $norm] = $this->cache[$param->id()];

        $vHat = $gradient->subtract($velocity)
            ->multiply($this->momentumDecay);

        $velocity = $velocity->add($vHat);

        $nHat = $gradient->square()->subtract($norm)
            ->multiply($this->normDecay);

        $norm = $norm->add($nHat);

        $this->cache[$param->id()] = [$velocity, $norm];

        $norm = $norm->sqrt()->clipLower(EPSILON);

        return $velocity->multiply($this->scheduler->rate())->divide($norm);
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
        return "Adam (scheduler: {$this->scheduler}, momentum decay: {$this->momentumDecay},"
            . " norm decay: {$this->normDecay})";
    }
}

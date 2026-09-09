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
 * RMS Prop
 *
 * An adaptive gradient technique that divides the current gradient over a rolling window
 * of magnitudes of recent gradients.
 *
 * References:
 * [1] T. Tieleman et al. (2012). Lecture 6e rmsprop: Divide the
 * gradient by a running average of its recent magnitude.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RMSProp implements Optimizer
{
    /**
     * The learning rate schedule.
     *
     * @var Scheduler
     */
    protected Scheduler $scheduler;

    /**
     * The rms decay rate.
     *
     * @var float
     */
    protected float $decay;

    /**
     * The opposite of the rms decay rate.
     *
     * @var float
     */
    protected float $rho;

    /**
     * The cache of running squared gradients.
     *
     * @var Tensor[]
     */
    protected array $cache = [
        //
    ];

    /**
     * @param Scheduler $scheduler
     * @param float $decay
     * @throws InvalidArgumentException
     */
    public function __construct(Scheduler $scheduler, float $decay = 0.1)
    {
        if ($decay <= 0.0 or $decay >= 1.0) {
            throw new InvalidArgumentException('Decay must be between'
                . " 0 and 1, $decay given.");
        }

        $this->scheduler = $scheduler;
        $this->decay = $decay;
        $this->rho = 1.0 - $decay;
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

        $norm = $norm->multiply($this->rho)
            ->add($gradient->square()->multiply($this->decay));

        $this->cache[$param->id()] = $norm;

        return $gradient->multiply($this->scheduler->rate())
            ->divide($norm->sqrt()->clipLower(EPSILON));
    }

    /**
     * Advance the paired learning-rate schedule by one batch.
     *
     * @internal
     */
    public function step() : void
    {
        $this->scheduler->tick();
    }

    /**
     * Reset the parameter cache.
     *
     * @internal
     */
    public function reset() : void
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
        return "RMS Prop (scheduler: {$this->scheduler}, decay: {$this->decay})";
    }
}

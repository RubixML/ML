<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Stringable;

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
class RMSProp implements Optimizer, Adaptive, Stringable
{
    /**
     * The learning rate that controls the global step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The rms decay rate.
     *
     * @var float
     */
    protected $decay;

    /**
     * The opposite of the rms decay rate.
     *
     * @var float
     */
    protected $rho;

    /**
     * The cache of running squared gradients.
     *
     * @var \Tensor\Tensor[]
     */
    protected $cache = [
        //
    ];

    /**
     * @param float $rate
     * @param float $decay
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $rate = 0.001, float $decay = 0.1)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($decay <= 0.0 or $decay >= 1.0) {
            throw new InvalidArgumentException('Decay must be between'
                . " 0 and 1, $decay given.");
        }

        $this->rate = $rate;
        $this->decay = $decay;
        $this->rho = 1.0 - $decay;
    }

    /**
     * Warm the parameter cache.
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     */
    public function warm(Parameter $param) : void
    {
        $this->cache[$param->id()] = get_class($param->param())::zeros(...$param->param()->shape());
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Tensor\Tensor<int|float|array> $gradient
     * @return \Tensor\Tensor<int|float|array>
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor
    {
        $norm = $this->cache[$param->id()];

        $norm = $norm->multiply($this->rho)
            ->add($gradient->square()->multiply($this->decay));

        $this->cache[$param->id()] = $norm;

        return $gradient->multiply($this->rate)
            ->divide($norm->sqrt()->clipLower(EPSILON));
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "RMS Prop (rate: {$this->rate}, decay: {$this->decay})";
    }
}

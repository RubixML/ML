<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use InvalidArgumentException;

/**
 * Adam
 *
 * Short for *Adaptive Moment Estimation*, the Adam Optimizer combines both
 * Momentum and RMS prop to achieve a balance of velocity and stability. In
 * addition to storing an exponentially decaying average of past squared
 * gradients like RMSprop, Adam also keeps an exponentially decaying average
 * of past gradients, similar to Momentum. Whereas Momentum can be seen as a
 * ball running down a slope, Adam behaves like a heavy ball with friction,
 * which thus prefers flat minima in the error surface.
 *
 * References:
 * [1] D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Adam implements Optimizer, Adaptive
{
    protected const WARM_UP_STEPS = 50;

    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The decay rate of the momentum property.
     *
     * @var float
     */
    protected $momentumDecay;

    /**
     * The decay rate of the RMS property.
     *
     * @var float
     */
    protected $rmsDecay;

    /**
     * The per parameter velocity and squared gradient cache.
     *
     * @var array[]
     */
    protected $cache = [
        //
    ];

    /**
     * The number of steps taken since initialization.
     *
     * @var int
     */
    protected $t = 0;

    /**
     * @param float $rate
     * @param float $momentumDecay
     * @param float $rmsDecay
     * @throws \InvalidArgumentException
     */
    public function __construct(float $rate = 0.001, float $momentumDecay = 0.9, float $rmsDecay = 0.999)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException('The learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($momentumDecay < 0. or $momentumDecay > 1.) {
            throw new InvalidArgumentException('Momentum decay must be between'
                . ' 0 and 1.');
        }

        if ($rmsDecay < 0. or $rmsDecay > 1.) {
            throw new InvalidArgumentException('RMS decay rate must be between'
                . ' 0 and 1.');
        }

        $this->rate = $rate;
        $this->momentumDecay = $momentumDecay;
        $this->rmsDecay = $rmsDecay;
    }

    /**
     * Initialize a parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     */
    public function initialize(Parameter $param) : void
    {
        $velocity = Matrix::zeros(...$param->w()->shape());
        $g2 = clone $velocity;

        $this->cache[$param->id()] = [$velocity, $g2];
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Rubix\Tensor\Matrix $gradient
     */
    public function step(Parameter $param, Matrix $gradient) : void
    {
        [$velocity, $g2] = $this->cache[$param->id()];

        $velocity = $velocity->multiply($this->momentumDecay)
            ->add($gradient->multiply(1. - $this->momentumDecay));

        $g2 = $g2->multiply($this->rmsDecay)
            ->add($gradient->square()->multiply(1. - $this->rmsDecay));

        $this->cache[$param->id()] = [$velocity, $g2];

        if ($this->t < self::WARM_UP_STEPS) {
            $this->t++;
            
            $velocity = $velocity->divide(1. - $this->momentumDecay ** $this->t);

            $g2 = $g2->divide(1. - $this->rmsDecay ** $this->t);
        }

        $step = $velocity->multiply($this->rate)
            ->divide($g2->sqrt()->clipLower(self::EPSILON));

        $param->update($step);
    }
}

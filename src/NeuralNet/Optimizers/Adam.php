<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use InvalidArgumentException;

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
     * The momentum decay rate.
     *
     * @var float
     */
    protected $momentumDecay;

    /**
     * The decay rate of the previous norms.
     *
     * @var float
     */
    protected $normDecay;

    /**
     * The opposite of the momentum decay.
     *
     * @var float
     */
    protected $beta1;

    /**
     * The opposite of the norm decay.
     *
     * @var float
     */
    protected $beta2;

    /**
     * The parameter cache of running velocity and squared gradients.
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
     * @param float $normDecay
     * @throws \InvalidArgumentException
     */
    public function __construct(float $rate = 0.001, float $momentumDecay = 0.1, float $normDecay = 0.001)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($momentumDecay <= 0. or $momentumDecay >= 1.) {
            throw new InvalidArgumentException('Momentum decay must be between'
                . " 0 and 1, $momentumDecay given.");
        }

        if ($normDecay <= 0. or $normDecay >= 1.) {
            throw new InvalidArgumentException('Norm decay must be between'
                . " 0 and 1, $normDecay given.");
        }

        $this->rate = $rate;
        $this->momentumDecay = $momentumDecay;
        $this->normDecay = $normDecay;
        $this->beta1 = 1. - $momentumDecay;
        $this->beta2 = 1. - $normDecay;
    }

    /**
     * Warm the cache.
     *
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter $param
     */
    public function warm(Parameter $param) : void
    {
        $velocity = get_class($param->w())::zeros(...$param->w()->shape());
        $norm = clone $velocity;

        $this->cache[$param->id()] = [$velocity, $norm];
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter $param
     * @param \Rubix\Tensor\Tensor $gradient
     * @return \Rubix\Tensor\Tensor
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor
    {
        [$velocity, $norm] = $this->cache[$param->id()];

        $velocity = $velocity->multiply($this->beta1)
            ->add($gradient->multiply($this->momentumDecay));

        $norm = $norm->multiply($this->beta2)
            ->add($gradient->square()->multiply($this->normDecay));

        $this->cache[$param->id()] = [$velocity, $norm];

        if ($this->t < self::WARM_UP_STEPS) {
            $this->t++;
            
            $velocity = $velocity->divide(1. - $this->beta1 ** $this->t);

            $norm = $norm->divide(1. - $this->beta2 ** $this->t);
        }

        return $velocity->multiply($this->rate)
            ->divide($norm->sqrt()->clipLower(EPSILON));
    }
}

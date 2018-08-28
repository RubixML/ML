<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use RuntimeException;

/**
 * Alpha Dropout
 *
 * Alpha Dropout is a type of dropout layer that maintains the mean and variance
 * of the original inputs in order to ensure the self-normalizing property of
 * SELU networks with dropout. Alpha Dropout fits with SELU networks by randomly
 * setting activations to the negative saturation value of the activation
 * function at a given ratio each pass.
 *
 * References:
 * [1] G. Klambauer et al. (2017). Self-Normalizing Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AlphaDropout implements Hidden, Nonparametric
{
    const ALPHA = 1.6732632423543772848170429916717;
    const SCALE = 1.0507009873554804934193349852946;

    const ALPHA_P = -self::ALPHA * self::SCALE;

    /**
     * The ratio of neurons that are dropped during each training pass.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The scaling coefficient.
     *
     * @var float
     */
    protected $a;

    /**
     * The centering coefficient.
     *
     * @var float
     */
    protected $b;

    /**
     * The width of the layer.
     *
     * @var int
     */
    protected $width;

    /**
     * The memoized dropout mask.
     *
     * @var \Rubix\ML\Other\Structures\Matrix|null
     */
    protected $mask;

    /**
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $ratio = 0.1)
    {
        if ($ratio <= 0. or $ratio >= 1.) {
            throw new InvalidArgumentException('Dropout ratio must be between 0'
                . ' and 1.0.');
        }

        $this->ratio = $ratio;
        $this->a = ((1. - $ratio) * (1. + $ratio * self::ALPHA_P ** 2)) ** -0.5;
        $this->b = -$this->a * self::ALPHA_P * $ratio;
        $this->width = 0;
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return $this->width;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int
    {
        $this->width = $fanIn;

        return $fanIn;
    }

    /**
     * Compute the input sum and activation of each neuron in the layer and
     * return an activation matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $m = $input->m();
        $n = $input->n();

        $mask = Matrix::zeros($m, $n)->map(function ($value) {
            return (rand(0, self::PHI) / self::PHI) > $this->ratio ? 1. : 0.;
        });

        $saturation = $mask->map(function ($value) {
            return $value === 0. ? self::ALPHA_P : 0.;
        });

        $activations = $input->multiply($mask)
            ->add($saturation)
            ->scalarMultiply($this->a)
            ->scalarAdd($this->b);

        $this->mask = $mask;

        return $activations;
    }

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $input;
    }

    /**
     * Calculate the errors and gradients of the layer and update the parameters.
     *
     * @param  callable  $prevErrors
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @throws \RuntimeException
     * @return callable
     */
    public function back(callable $prevErrors, Optimizer $optimizer) : callable
    {
        if (is_null($this->mask)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $errors = $prevErrors()->multiply($this->mask);

        unset($this->mask);

        return function () use ($errors) {
            return $errors;
        };
    }
}

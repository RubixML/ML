<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;

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
class AlphaDropout extends Dropout
{
    const ALPHA = 1.6732632423543772848170429916717;
    const SCALE = 1.0507009873554804934193349852946;

    const ALPHA_P = -self::ALPHA * self::SCALE;

    /**
     * The scaling coefficient.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The centering coefficient.
     *
     * @var float
     */
    protected $beta;

    /**
     * @param  float  $ratio
     * @return void
     */
    public function __construct(float $ratio = 0.1)
    {
        $this->alpha = ((1. - $ratio) * (1. + $ratio * self::ALPHA_P ** 2)) ** -0.5;
        $this->beta = -$this->alpha * self::ALPHA_P * $ratio;

        parent::__construct($ratio);
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @return \Rubix\Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $mask = Matrix::rand(...$input->shape())
            ->map([$this, 'drop']);

        $saturation = $mask->map([$this, 'saturate']);

        $this->mask = $mask;

        return $input->multiply($mask)
            ->add($saturation)
            ->multiply($this->alpha)
            ->add($this->beta);
    }

    /**
     * Boost dropped neurons by a factor of alpha p.
     * 
     * @param  float  $value
     * @return float
     */
    public function saturate(float $value) : float
    {
        return $value === 0. ? self::ALPHA_P : 0.;
    }
}

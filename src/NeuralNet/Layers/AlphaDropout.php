<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\SELU;

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
    /**
     * The negative saturation value of SELU.
     *
     * @var float
     */
    protected const ALPHA_P = -SELU::ALPHA * SELU::SCALE;

    /**
     * The affine transformation scaling coefficient.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The affine transformation centering coefficient.
     *
     * @var float
     */
    protected $beta;

    /**
     * @param float $ratio
     */
    public function __construct(float $ratio = 0.1)
    {
        parent::__construct($ratio);

        $this->alpha = ((1.0 - $ratio) * (1.0 + $ratio * self::ALPHA_P ** 2)) ** -0.5;
        $this->beta = -$this->alpha * self::ALPHA_P * $ratio;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $mask = Matrix::rand(...$input->shape())
            ->greater($this->ratio);

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
     * @param int $value
     * @return float
     */
    public function saturate(int $value) : float
    {
        return $value === 0 ? self::ALPHA_P : 0.0;
    }
}

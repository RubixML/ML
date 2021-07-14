<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Vector;
use Tensor\Matrix;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\NeuralNet\Parameter;

use const Rubix\ML\EPSILON;

/**
 * AdaMax
 *
 * A version of Adam that replaces the RMS property with the infinity norm of the gradients.
 *
 * References:
 * [1] D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AdaMax extends Adam
{
    /**
     * Return the element-wise maximum of two tensors.
     *
     * @param \Tensor\Tensor $a
     * @param \Tensor\Tensor $b
     * @return \Tensor\Tensor
     */
    protected static function maximum(Tensor $a, Tensor $b) : Tensor
    {
        if ($a instanceof Matrix and $b instanceof Matrix) {
            $c = [];

            foreach ($a as $i => $valueA) {
                $c[] = static::maximum($valueA, $b[$i])->asArray();
            }

            return Matrix::quick($c);
        }

        $bHat = $b->asArray();

        $c = [];

        foreach ($a as $i => $valueA) {
            $c[] = (float) max($valueA, $bHat[$i]);
        }

        return Vector::quick($c);
    }

    /**
     * @param float $rate
     * @param float $momentumDecay
     * @param float $normDecay
     */
    public function __construct(float $rate = 0.001, float $momentumDecay = 0.1, float $normDecay = 0.001)
    {
        if (ExtensionIsLoaded::with('tensor')->passes()) {
            ExtensionMinimumVersion::with('tensor', '3.0.0-beta')->check();
        }

        parent::__construct($rate, $momentumDecay, $normDecay);
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @internal
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Tensor\Tensor<int|float|array> $gradient
     * @return \Tensor\Tensor<int|float|array>
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor
    {
        [$velocity, $norm] = $this->cache[$param->id()];

        $velocity = $velocity->multiply(1.0 - $this->momentumDecay)
            ->add($gradient->multiply($this->momentumDecay));

        $norm = $norm->multiply(1.0 - $this->normDecay);

        $norm = static::maximum($norm, $gradient->abs());

        $this->cache[$param->id()] = [$velocity, $norm];

        $norm = $norm->clipLower(EPSILON);

        return $velocity->divide($norm)->multiply($this->rate);
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
        return "AdaMax (rate: {$this->rate}, momentum_decay: {$this->momentumDecay},"
            . " norm_decay: {$this->normDecay})";
    }
}

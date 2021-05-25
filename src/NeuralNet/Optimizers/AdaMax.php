<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Vector;
use Tensor\Matrix;
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

            foreach ($a->asVectors() as $i => $valueA) {
                $c[] = static::maximum($valueA, $b->rowAsVector($i))->asArray();
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

        $velocity = $velocity->multiply($this->beta1)
            ->add($gradient->multiply($this->momentumDecay));

        $norm = $norm->multiply($this->beta2);

        $norm = static::maximum($norm, $gradient->abs());

        $this->cache[$param->id()] = [$velocity, $norm];

        $norm = $norm->clipLower(EPSILON);

        return $velocity->divide($norm)->multiply($this->rate);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "AdaMax (rate: {$this->rate}, momentum decay: {$this->momentumDecay},"
            . " norm decay: {$this->normDecay})";
    }
}

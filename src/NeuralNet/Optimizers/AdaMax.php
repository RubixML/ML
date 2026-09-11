<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\Exceptions\RuntimeException;
use Tensor\Vector;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;

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
     * @param Tensor $a
     * @param Tensor $b
     * @return Tensor
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
     * @param Scheduler $scheduler
     * @param float $momentumDecay
     * @param float $normDecay
     */
    public function __construct(Scheduler $scheduler, float $momentumDecay = 0.1, float $normDecay = 0.001)
    {
        if (ExtensionIsLoaded::with('tensor')->passes()) {
            ExtensionMinimumVersion::with('tensor', '3.0.0-beta')->check();
        }

        parent::__construct($scheduler, $momentumDecay, $normDecay);
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @internal
     *
     * @param Parameter $param
     * @return Tensor<int|float|array>
     */
    public function update(Parameter $param) : Tensor
    {
        if (!$param->hasGradient()) {
            throw new RuntimeException('Cannot update parameter with no gradient.');
        }

        [$velocity, $norm] = $this->cache[$param->id()];

        $vHat = $param->gradient()->subtract($velocity)
            ->multiply($this->momentumDecay);

        $velocity = $velocity->add($vHat);

        $norm = $norm->multiply(1.0 - $this->normDecay);

        $norm = static::maximum($norm, $param->gradient()->abs());

        $this->cache[$param->id()] = [$velocity, $norm];

        $norm = $norm->clipLower(EPSILON);

        $step = $velocity->divide($norm)->multiply($this->scheduler->rate());

        return $step;
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
        return "AdaMax (scheduler: {$this->scheduler}, momentum_decay: {$this->momentumDecay},"
            . " norm_decay: {$this->normDecay})";
    }
}

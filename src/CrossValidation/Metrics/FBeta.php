<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;

use const Rubix\ML\EPSILON;

/**
 * F Beta
 *
 * A weighted harmonic mean of precision and recall. The beta parameter controls the
 * weight of precision in the combined score. As beta goes to infinity the score
 * only considers recall whereas when it goes to 0 it only considers precision. When
 * beta is equal to 1, the metric is called an F1 score.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class FBeta implements Metric
{
    /**
     * The squared weight of precision in the harmonic mean.
     *
     * @var float
     */
    protected $beta2;

    /**
     * Compute the class precision.
     *
     * @param int $tp
     * @param int $fp
     * @return float
     */
    public static function precision(int $tp, int $fp) : float
    {
        return $tp / (($tp + $fp) ?: EPSILON);
    }

    /**
     * Compute the class recall.
     *
     * @param int $tp
     * @param int $fn
     * @return float
     */
    public static function recall(int $tp, int $fn) : float
    {
        return $tp / (($tp + $fn) ?: EPSILON);
    }

    /**
     * @param float $beta
     * @throws \InvalidArgumentException
     */
    public function __construct(float $beta = 1.)
    {
        if ($beta < 0.) {
            throw new InvalidArgumentException('Beta cannot be less'
                . " than 0, $beta given.");
        }

        $this->beta2 = $beta ** 2;
    }

    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0., 1.];
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            Estimator::CLASSIFIER,
            Estimator::ANOMALY_DETECTOR,
        ];
    }

    /**
     * Score a set of predictions.
     *
     * @param array $predictions
     * @param array $labels
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        if (empty($predictions)) {
            return 0.;
        }

        $classes = array_unique(array_merge($predictions, $labels));

        $k = count($classes);

        $truePos = $falsePos = $falseNeg = array_fill_keys($classes, 0);

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            if ($prediction == $label) {
                ++$truePos[$prediction];
            } else {
                ++$falsePos[$prediction];
                ++$falseNeg[$label];
            }
        }

        $precision = Stats::mean(
            array_map([self::class, 'precision'], $truePos, $falsePos)
        );

        $recall = Stats::mean(
            array_map([self::class, 'recall'], $truePos, $falseNeg)
        );

        return (1. + $this->beta2) * $precision * $recall
            / (($this->beta2 * $precision + $recall) ?: EPSILON);
    }
}

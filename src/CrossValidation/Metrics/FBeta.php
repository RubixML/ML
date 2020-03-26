<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;

use function count;

use const Rubix\ML\EPSILON;

/**
 * F-Beta
 *
 * A weighted harmonic mean of precision and recall, F-Beta is a both a versatile and balanced
 * metric. The beta parameter controls the weight of precision in the combined score. As beta
 * goes to infinity the score only considers recall, whereas when it goes to 0 it only
 * considers precision. When beta is equal to 1, the metric is called an F1 score.
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
    public function __construct(float $beta = 1.0)
    {
        if ($beta < 0.0) {
            throw new InvalidArgumentException('Beta must be'
                . " greater than 0, $beta given.");
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
        return [0.0, 1.0];
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @return \Rubix\ML\EstimatorType[]
     */
    public function compatibility() : array
    {
        return [
            EstimatorType::classifier(),
            EstimatorType::anomalyDetector(),
        ];
    }

    /**
     * Score a set of predictions.
     *
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('Number of predictions'
                . ' and labels must be equal.');
        }

        if (empty($predictions)) {
            return 0.0;
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

        return (1.0 + $this->beta2) * $precision * $recall
            / (($this->beta2 * $precision + $recall) ?: EPSILON);
    }
}

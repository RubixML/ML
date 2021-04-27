<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function array_unique;
use function array_merge;
use function array_fill_keys;

use const Rubix\ML\EPSILON;

/**
 * F-Beta
 *
 * A weighted harmonic mean of precision and recall, F-Beta is a both a versatile and balanced
 * metric. The beta parameter controls the weight of precision in the combined score. As beta
 * goes to infinity the score only considers recall, whereas when it goes to 0 it only
 * considers precision. When beta is equal to 1, this metric is called an F1 score.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class FBeta implements Metric
{
    /**
     * The ratio of weight given precision over recall.
     *
     * @var float
     */
    protected $beta;

    /**
     * Compute the class precision.
     *
     * @internal
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
     * @internal
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
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $beta = 1.0)
    {
        if ($beta < 0.0) {
            throw new InvalidArgumentException('Beta must be'
                . " greater than 0, $beta given.");
        }

        $this->beta = $beta;
    }

    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return array{float,float}
     */
    public function range() : array
    {
        return [0.0, 1.0];
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\EstimatorType>
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
     * @param list<string|int> $predictions
     * @param list<string|int> $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        PredictionAndLabelCountsAreEqual::with($predictions, $labels)->check();

        if (empty($predictions)) {
            return 0.0;
        }

        $classes = array_unique(array_merge($predictions, $labels));

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

        $precision = Stats::mean(array_map([self::class, 'precision'], $truePos, $falsePos));

        $recall = Stats::mean(array_map([self::class, 'recall'], $truePos, $falseNeg));

        return (1.0 + $this->beta ** 2) * $precision * $recall
            / (($this->beta ** 2 * $precision + $recall) ?: EPSILON);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "F Beta (beta: {$this->beta})";
    }
}

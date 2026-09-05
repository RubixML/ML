<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;
use Rubix\ML\Set;

use const Rubix\ML\EPSILON;

/**
 * Informedness
 *
 * Informedness a multiclass generalization of Youden's J Statistic and can be interpreted as the
 * probability that an estimator will make an informed prediction. Its value ranges from -1 through
 * 1 and has a value of 0 when the test yields no useful information.
 *
 * References:
 * [1] W. J. Youden. (1950). Index for Rating Diagnostic Tests.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Informedness implements Metric
{
    /**
     * Compute the class Informedness score.
     *
     * @internal
     *
     * @param int $tp
     * @param int $tn
     * @param int $fp
     * @param int $fn
     * @return float
     */
    public static function compute(int $tp, int $tn, int $fp, int $fn) : float
    {
        return $tp / (($tp + $fn) ?: EPSILON) + $tn / (($tn + $fp) ?: EPSILON) - 1.0;
    }

    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return Tuple<float,float>
     */
    public function range() : Tuple
    {
        return new Tuple(-1.0, 1.0);
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @internal
     *
     * @return list<EstimatorType>
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

        $classes = new Set(...$predictions, ...$labels);

        $classes = $classes->toArray();

        $n = count($predictions);

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

        $trueNeg = [];

        foreach ($classes as $class) {
            $trueNeg[$class] = $n - $truePos[$class] - $falsePos[$class] - $falseNeg[$class];
        }

        $scores = array_map([self::class, 'compute'], $truePos, $trueNeg, $falsePos, $falseNeg);

        return Stats::mean($scores);
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
        return 'Informedness';
    }
}

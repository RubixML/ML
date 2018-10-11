<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

/**
 * Informedness
 *
 * Informedness is a measure of the probability that an estimator will make an
 * informed decision. The index was suggested by W.J. Youden as a way of
 * summarizing the performance of a diagnostic test. Its value ranges from 0
 * through 1 and has a zero value when the test gives the same proportion of
 * positive results for groups with and without the disease, i.e the test is
 * useless.
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
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0., 1.];
    }

    /**
     * Calculate the informedness score of the predicted classes. Informedness
     * is determined by recall + specificity - 1.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(Estimator $estimator, Dataset $testing) : float
    {
        if ($estimator->type() !== Estimator::CLASSIFIER and $estimator->type() !== Estimator::DETECTOR) {
            throw new InvalidArgumentException('This metric only works with'
                . ' classifiers and anomaly detectors.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This metric requires a labeled'
                . ' testing set.');
        }

        $n = $testing->numRows();

        if ($n === 0) {
            return 0.;
        }

        $predictions = $estimator->predict($testing);

        $labels = $testing->labels();

        $classes = array_unique(array_merge($predictions, $labels));

        $k = count($classes);

        $truePositives = $trueNegatives = $falsePositives = $falseNegatives
            = array_fill_keys($classes, 0);

        foreach ($predictions as $i => $outcome) {
            if ($outcome === $labels[$i]) {
                $truePositives[$outcome]++;

                foreach ($classes as $class) {
                    if ($class !== $outcome) {
                        $trueNegatives[$class]++;
                    }
                }
            } else {
                $falsePositives[$outcome]++;
                $falseNegatives[$labels[$i]]++;
            }
        }

        $score = 0.;

        foreach ($truePositives as $class => $tp) {
            $tn = $trueNegatives[$class];
            $fp = $falsePositives[$class];
            $fn = $falseNegatives[$class];

            $score += ($tp + self::EPSILON) / ($tp + $fn + self::EPSILON)
                + ($tn + self::EPSILON) / ($tn + $fp + self::EPSILON)
                - 1.;
        }

        return $score / $k;
    }
}

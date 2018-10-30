<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

/**
 * MCC
 *
 * Matthews Correlation Coefficient measures the quality of a classification. It
 * takes into account true and false positives and negatives and is generally
 * regarded as a balanced measure which can be used even if the classes are of
 * very different sizes. The MCC is in essence a correlation coefficient between
 * the observed and predicted binary classifications; it returns a value between
 * −1 and +1. A coefficient of +1 represents a perfect prediction, 0 no better
 * than random prediction and −1 indicates total disagreement between prediction
 * and observation.
 *
 * References:
 * [1] B. W. Matthews. (1975). Comparison of the Predicted and Observed Secondary
 * Structure of T4 Phage Lysozyme.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MCC implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-1., 1.];
    }

    /**
     * Score the Matthews correlation coefficient of the predicted classes.
     * Score is a number between -1 and 1.
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

        if ($testing->numRows() < 1) {
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

            $score += ($tp * $tn - $fp * $fn)
                / sqrt((($tp + $fp) * ($tp + $fn) * ($tn + $fp) * ($tn + $fn)) ?: self::EPSILON);
        }

        return $score / $k;
    }
}

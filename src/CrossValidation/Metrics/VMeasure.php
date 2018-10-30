<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

/**
 * V Measure
 *
 * V Measure is the harmonic balance between homogeneity and completeness
 * and is used as a measure to determine the quality of a clustering.
 * 
 * References:
 * [1] A. Rosenberg et al. (2007). V-Measure: A conditional entropy-based
 * external cluster evaluation measure.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class VMeasure implements Metric
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
     * Calculate the V score of a clustering. V Score is the harmonic mean of
     * homogeneity and completness.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(Estimator $estimator, Dataset $testing) : float
    {
        if ($estimator->type() !== Estimator::CLUSTERER) {
            throw new InvalidArgumentException('This metric only works with'
                . ' clusterers.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This metric requires a labeled'
                . ' testing set.');
        }

        if ($testing->numRows() < 0) {
            return 0.;
        }

        $predictions = $estimator->predict($testing);

        $labels = $testing->labels();

        $clusters = array_unique($predictions);
        $classes = array_unique($labels);

        $table = [[], []];

        foreach ($clusters as $outcome) {
            $table[0][$outcome] = array_fill_keys($classes, 0);
        }

        foreach ($classes as $label) {
            $table[1][$label] = array_fill_keys($clusters, 0);
        }

        foreach ($predictions as $i => $outcome) {
            $table[0][$outcome][$labels[$i]] += 1;
        }

        foreach ($labels as $i => $class) {
            $table[1][$class][$predictions[$i]] += 1;
        }

        $homogeneity = $completeness = 0.;

        foreach ($table[0] as $distribution) {
            $homogeneity += max($distribution)
                / (array_sum($distribution) ?: self::EPSILON);
        }

        foreach ($table[1] as $distribution) {
            $completeness += max($distribution)
                / (array_sum($distribution) ?: self::EPSILON);
        }

        $homogeneity /= count($clusters);
        $completeness /= count($classes);

        return 2. * ($homogeneity * $completeness)
            / ($homogeneity + $completeness);
    }
}

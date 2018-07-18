<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use Rubix\ML\Clusterers\Clusterer;
use InvalidArgumentException;

class Concentration implements Validation
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return array
     */
    public function range() : array
    {
        return [-INF, INF];
    }

    /**
     * Calculate the Calinski Harabaz score of a clustering. The score
     * is defined as ratio between the within-cluster dispersion and the
     * between-cluster dispersion.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(Estimator $estimator, Dataset $testing) : float
    {
        if (!$estimator instanceof Clusterer) {
            throw new InvalidArgumentException('This metric only works on'
                . ' clusterers.');
        }

        $predictions = $estimator->predict($testing);

        $labeled = new Labeled($testing->samples(), $predictions);

        $globals = array_map(function ($values) {
            return Average::mean($values);
        }, $testing->rotate());

        $intra = $extra = 0.0;

        $strata = $labeled->stratify();

        foreach ($strata as $cluster => $dataset) {
            $centroid = [];

            foreach ($dataset->rotate() as $column => $values) {
                $centroid[$column] = Average::mean((array) $values);
            }

            foreach ($dataset as $row => $sample) {
                foreach ($sample as $column => $feature) {
                    $intra += ($feature - $centroid[$column]) ** 2;
                }
            }

            $temp = 0.0;

            foreach ($centroid as $column => $mean) {
                $temp += ($mean - $globals[$column]) ** 2;
            }

            $temp *= $dataset->numRows();

            $extra += $temp;
        }

        if ($intra === 0.0) {
            return 1.0;
        }

        return ($extra * ($testing->numRows() - count($strata)))
            / ($intra * (count($strata) - 1) + self::EPSILON);
    }
}

<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\ProbabilitiesAreNotEmpty;
use Rubix\ML\Specifications\LabelsAreCompatibleWithProbabilities;

/**
 * Probabilistic distance
 *
 * Distance = 1 – The Probability for the outcome if it comes up
 *
 * **Note:** In order to maintain the convention of *maximizing* validation scores,
 * this metric outputs the negative of the original score.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Alex Torchenko
 */
class ProbabilisticDistance implements ProbabilisticMetric
{
    public function __toString(): string
    {
        return 'Probabilistic distance';
    }

    /**
     * {@inheritDoc}
     */
    public function range(): Tuple
    {
        return new Tuple(-1.0, 0.0);
    }

    public function score(array $probabilities, array $labels): float
    {
        SpecificationChain::with([
            ProbabilitiesAreNotEmpty::with($probabilities),
            LabelsAreCompatibleWithProbabilities::with($probabilities, $labels),
        ])->check();

        $distances = [];

        foreach ($probabilities as $i => $row) {
            foreach ($row as $class => $probability) {
                if ($class === $labels[$i]) {
                    $distances[] = -(1 - $probability);
                    break;
                }
            }
        }

        return array_sum($distances) / count($distances);
    }
}

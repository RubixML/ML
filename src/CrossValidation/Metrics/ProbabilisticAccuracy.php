<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\Specifications\ProbabilityAndLabelCountsAreEqual;

/**
 * Probabilistic Accuracy
 *
 * This metric comes from the sports betting domain, where it's used to measure the accuracy of
 * predictions by looking at the probabilities of class predictions. Accordingly, this metric places
 * additional weight on the "confidence" of each prediction.
 *
 * !!! note
 *     Metric assumes probabilities are between 0 and 1 and their joint distribution sums to 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Alex Torchenko
 */
class ProbabilisticAccuracy implements ProbabilisticMetric
{
    /**
     * {@inheritDoc}
     */
    public function range() : Tuple
    {
        return new Tuple(0.0, 1.0);
    }

    /**
     * Return the validation score of a set of probabilities with their ground-truth labels.
     *
     * @param list<array<string|int,float>> $probabilities
     * @param list<string|int> $labels
     * @return float
     */
    public function score(array $probabilities, array $labels) : float
    {
        ProbabilityAndLabelCountsAreEqual::with($probabilities, $labels)->check();

        $n = count($probabilities);

        if ($n === 0) {
            return 0.0;
        }

        $score = 0.0;

        foreach ($probabilities as $i => $dist) {
            $score += $dist[$labels[$i]] ?? 0.0;
        }

        return $score / $n;
    }

    /**
     * {@inheritDoc}
     */
    public function __toString() : string
    {
        return 'Probabilistic Accuracy';
    }
}

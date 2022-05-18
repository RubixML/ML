<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\Specifications\ProbabilityAndLabelCountsAreEqual;

/**
 * Brier Score
 *
 * Brier Score is a *strictly proper* scoring metric that is equivalent to applying mean squared
 * error to the probabilities of a probabilistic estimator.
 *
 * !!! note
 *     Metric assumes probabilities are between 0 and 1 and their joint distribution sums to 1.
 *
 * References:
 * [1] G. W. Brier. (1950). Verification of Forecasts Expresses in Terms of Probability
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BrierScore implements ProbabilisticMetric
{
    /**
     * {@inheritDoc}
     */
    public function range() : Tuple
    {
        return new Tuple(-2.0, 0.0);
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

        $error = 0.0;

        foreach ($probabilities as $i => $dist) {
            $label = $labels[$i];

            foreach ($dist as $class => $probability) {
                $expected = $class == $label ? 1.0 : 0.0;

                $error += ($probability - $expected) ** 2;
            }
        }

        $error /= $n;

        return -$error;
    }

    /**
     * {@inheritDoc}
     */
    public function __toString() : string
    {
        return 'Brier Score';
    }
}

<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Other\Specifications\EstimatorIsCompatibleWithMetric;
use InvalidArgumentException;

/**
 * Leave P Out
 *
 * Leave P Out cross-validation involves using p observations as the validation
 * set and the remaining observations as the training set. This is repeated on
 * all ways to cut the original sample on a validation set of p observations
 * and a training set. The resulting score is an average of all the tests.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LeavePOut implements Validator
{
    /**
     * The number of samples to leave out each round for testing.
     *
     * @var int
     */
    protected $p;

    /**
     * @param int $p
     * @throws \InvalidArgumentException
     */
    public function __construct(int $p = 10)
    {
        if ($p < 1) {
            throw new InvalidArgumentException('P cannot be less'
                . "than 1, $p given.");
        }

        $this->p = $p;
    }

    /**
     * Test the estimator with the supplied dataset and return a score.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @throws \InvalidArgumentException
     * @return float
     */
    public function test(Learner $estimator, Labeled $dataset, Metric $metric) : float
    {
        EstimatorIsCompatibleWithMetric::check($estimator, $metric);

        $n = (int) round($dataset->numRows() / $this->p);

        $score = 0.;

        for ($i = 0; $i < $n; $i++) {
            $training = clone $dataset;

            $testing = $training->splice($i * $this->p, $this->p);

            $estimator->train($training);

            $predictions = $estimator->predict($testing);

            $score += $metric->score($predictions, $testing->labels());
        }

        return $score / $n;
    }
}

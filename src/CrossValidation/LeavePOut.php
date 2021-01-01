<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Learner;
use Rubix\ML\Parallel;
use Rubix\ML\Estimator;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Backends\Tasks\TrainAndValidate;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Leave P Out
 *
 * Leave P Out tests a learner with a unique holdout set of size p for each iteration until
 * all samples have been tested. Although Leave P Out can take long with large datasets and
 * small values of p, it is especially suited for small datasets.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LeavePOut implements Validator, Parallel
{
    use Multiprocessing;

    /**
     * The number of samples to leave out each round for testing.
     *
     * @var int
     */
    protected $p;

    /**
     * @param int $p
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $p = 10)
    {
        if ($p < 1) {
            throw new InvalidArgumentException('P must be greater'
                . " than 0, $p given.");
        }

        $this->p = $p;
        $this->backend = new Serial();
    }

    /**
     * Test the estimator with the supplied dataset and return a validation score.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float
     */
    public function test(Learner $estimator, Labeled $dataset, Metric $metric) : float
    {
        EstimatorIsCompatibleWithMetric::with($estimator, $metric)->check();

        $n = (int) round($dataset->numRows() / $this->p);

        $this->backend->flush();

        for ($i = 0; $i < $n; ++$i) {
            $training = clone $dataset;

            $testing = $training->splice($i * $this->p, $this->p);

            $this->backend->enqueue(
                new TrainAndValidate($estimator, $training, $testing, $metric)
            );
        }

        $scores = $this->backend->process();

        return Stats::mean($scores);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Leave P Out (p: {$this->p})";
    }
}

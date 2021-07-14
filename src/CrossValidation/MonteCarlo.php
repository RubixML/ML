<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Learner;
use Rubix\ML\Parallel;
use Rubix\ML\Estimator;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\Multiprocessing;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Backends\Tasks\TrainAndValidate;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Monte Carlo
 *
 * Monte Carlo cross validation (or *repeated random subsampling*) is a technique that
 * averages the validation score of a learner over a user-defined number of simulations
 * where the learner is trained and tested on random splits of the dataset. The estimated
 * validation score approaches the actual validation score as the number of simulations
 * goes to infinity, however, only a tiny fraction of all possible simulations are needed
 * to produce a pretty good approximation.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MonteCarlo implements Validator, Parallel
{
    use Multiprocessing;

    /**
     * The number of simulations i.e. random subsamplings of the dataset.
     *
     * @var int
     */
    protected int $simulations;

    /**
     * The hold out ratio. i.e. the ratio of samples to use for testing.
     *
     * @var float
     */
    protected float $ratio;

    /**
     * @param int $simulations
     * @param float $ratio
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $simulations = 10, float $ratio = 0.2)
    {
        if ($simulations < 1) {
            throw new InvalidArgumentException('Number of simulations'
                . " must be greater than 0, $simulations given.");
        }

        if ($ratio <= 0.0 or $ratio >= 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        $this->simulations = $simulations;
        $this->ratio = $ratio;
        $this->backend = new Serial();
    }

    /**
     * Test the estimator with the supplied dataset and return a validation score.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return float
     */
    public function test(Learner $estimator, Labeled $dataset, Metric $metric) : float
    {
        EstimatorIsCompatibleWithMetric::with($estimator, $metric)->check();

        if ($dataset->numSamples() * $this->ratio < 1) {
            throw new RuntimeException('Dataset does not contain'
                . ' enough records to create a validation set with a'
                . " hold out ratio of {$this->ratio}.");
        }

        $stratify = $dataset->labelType()->isCategorical();

        $this->backend->flush();

        for ($i = 0; $i < $this->simulations; ++$i) {
            $dataset->randomize();

            [$testing, $training] = $stratify
                ? $dataset->stratifiedSplit($this->ratio)
                : $dataset->split($this->ratio);

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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Monte Carlo (simulations: {$this->simulations}, ratio: {$this->ratio})";
    }
}

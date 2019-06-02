<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Learner;
use Rubix\ML\Deferred;
use Rubix\ML\Parallel;
use Rubix\ML\Estimator;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Other\Specifications\EstimatorIsCompatibleWithMetric;
use InvalidArgumentException;

/**
 * Monte Carlo
 *
 * Repeated Random Subsampling or Monte Carlo cross validation is a technique
 * that takes the average validation score over a user-supplied number of
 * simulations (random splits of the dataset).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MonteCarlo implements Validator, Parallel
{
    use Multiprocessing;

    /**
     * The number of simulations to run i.e the number of tests to average.
     *
     * @var int
     */
    protected $simulations;

    /**
     * The hold out ratio. i.e. the ratio of samples to use for testing.
     *
     * @var float
     */
    protected $ratio;

    /**
     * Should we stratify the dataset before splitting?
     *
     * @var bool
     */
    protected $stratify;

    /**
     * @param int $simulations
     * @param float $ratio
     * @param bool $stratify
     * @throws \InvalidArgumentException
     */
    public function __construct(int $simulations = 10, float $ratio = 0.2, bool $stratify = false)
    {
        if ($simulations < 2) {
            throw new InvalidArgumentException('Must run at least 2'
                . " simulations, $simulations given.");
        }

        if ($ratio <= 0.01 or $ratio >= 1.) {
            throw new InvalidArgumentException('Holdout ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        $this->simulations = $simulations;
        $this->ratio = $ratio;
        $this->stratify = $stratify;
        $this->backend = new Serial();
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

        $this->backend->flush();

        for ($i = 0; $i < $this->simulations; $i++) {
            $dataset->randomize();

            [$testing, $training] = $this->stratify
                ? $dataset->stratifiedSplit($this->ratio)
                : $dataset->split($this->ratio);
    
            $this->backend->enqueue(new Deferred(
                [self::class, 'scorer'],
                [$estimator, $training, $testing, $metric]
            ));
        }
    
        $scores = $this->backend->process();

        return Stats::mean($scores);
    }

    /**
     * Score an estimator on one of n simulations.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Dataset $training
     * @param \Rubix\ML\Datasets\Labeled $testing
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @return float
     */
    public static function scorer(Learner $estimator, Dataset $training, Labeled $testing, Metric $metric) : float
    {
        $estimator->train($training);

        $predictions = $estimator->predict($testing);

        return $metric->score($predictions, $testing->labels());
    }
}

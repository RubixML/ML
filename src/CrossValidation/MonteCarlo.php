<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Learner;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\CrossValidation\Metrics\Metric;
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
class MonteCarlo implements Validator
{
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
     * The validation score of each simulation since the last test.
     * 
     * @var array|null
     */
    protected $scores;

    /**
     * @param  int  $simulations
     * @param  float  $ratio
     * @param  bool  $stratify
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $simulations = 10, float $ratio = 0.2, bool $stratify = false)
    {
        if ($simulations < 2) {
            throw new InvalidArgumentException('Must run at least 2'
                . " simulations, $simulations given.");
        }

        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException('Holdout ratio must be'
                . " between 0.01 and 1, $ratio given.");
        }

        $this->simulations = $simulations;
        $this->ratio = $ratio;
        $this->stratify = $stratify;
    }

    /**
     * Return the validation scores computed at last test time.
     * 
     * @return array|null
     */
    public function scores() : ?array
    {
        return $this->scores;
    }

    /**
     * Test the estimator with the supplied dataset and return a score.
     *
     * @param  \Rubix\ML\Learner  $estimator
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  \Rubix\ML\CrossValidation\Metrics\Metric  $metric
     * @return float
     */
    public function test(Learner $estimator, Labeled $dataset, Metric $metric) : float
    {
        $scores = [];

        for ($epoch = 0; $epoch < $this->simulations; $epoch++) {
            $dataset->randomize();

            if ($this->stratify) {
                list($testing, $training) = $dataset->stratifiedSplit($this->ratio);
            } else {
                list($testing, $training) = $dataset->split($this->ratio);
            }

            $estimator->train($training);

            $scores[] = $metric->score($estimator, $testing);
        }

        $this->scores = $scores;

        return Stats::mean($scores);
    }
}

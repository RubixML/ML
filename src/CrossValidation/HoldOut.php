<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Learner;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\Metric;
use InvalidArgumentException;

/**
 * Hold Out
 *
 * In the holdout method, we randomly assign data points to two sets (training
 * and testing set). The size of each of the testing set is given by the holdout
 * ratio and is typically smaller than the training set.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class HoldOut implements Validator
{
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
     * The validation score of the last test.
     * 
     * @var array|null
     */
    protected $scores;

    /**
     * @param  float  $ratio
     * @param  bool  $stratify
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $ratio = 0.2, bool $stratify = false)
    {
        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException('Holdout ratio must be'
                . " between 0.01 and 1, $ratio given.");
        }

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
        if ($this->stratify) {
            list($testing, $training) = $dataset->stratifiedSplit($this->ratio);
        } else {
            list($testing, $training) = $dataset->randomize()->split($this->ratio);
        }

        $estimator->train($training);

        $score = $metric->score($estimator, $testing);

        $this->scores = [$score];

        return $score;
    }
}

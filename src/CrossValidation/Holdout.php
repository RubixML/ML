<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\Validation;
use InvalidArgumentException;

class HoldOut implements Validator
{
    /**
     * The holdout ratio. i.e. the ratio of samples to use for testing.
     *
     * @var float
     */
    protected $ratio;

    /**
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $ratio = 0.2)
    {
        if ($ratio < 0.01 or $ratio > 1.0) {
            throw new InvalidArgumentException('Holdout ratio must be'
                . ' between 0.01 and 1.0.');
        }

        $this->ratio = $ratio;
    }

    /**
     * Run a single training and testing round where the ratio determines the
     * number of samples held out for testing.
     *
     * @param  \Rubix\ML\Estimator\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  \Rubix\ML\CrossValidation\Metrics\Validation  $metric
     * @return float
     */
    public function test(Estimator $estimator, Labeled $dataset, Validation $metric) : float
    {
        if ($estimator instanceof Classifier or $estimator instanceof Clusterer) {
            list($training, $testing) =
                $dataset->stratifiedSplit(1 - $this->ratio);
        } else {
            list($training, $testing) =
                $dataset->split(1 - $this->ratio);
        }

        $estimator->train($training);

        $score = $metric->score($estimator, $testing);

        return $score;
    }
}

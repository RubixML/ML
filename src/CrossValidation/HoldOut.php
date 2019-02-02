<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Other\Specifications\EstimatorIsCompatibleWithMetric;
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
     * Test the estimator with the supplied dataset and return a score.
     *
     * @param  \Rubix\ML\Learner  $estimator
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  \Rubix\ML\CrossValidation\Metrics\Metric  $metric
     * @throws \InvalidArgumentException
     * @return float
     */
    public function test(Learner $estimator, Labeled $dataset, Metric $metric) : float
    {
        EstimatorIsCompatibleWithMetric::check($estimator, $metric);

        [$testing, $training] = $this->stratify
            ? $dataset->stratifiedSplit($this->ratio)
            : $dataset->split($this->ratio);

        $estimator->train($training);

        $predictions = $estimator->predict($testing);

        return $metric->score($predictions, $testing->labels());
    }
}

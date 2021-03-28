<?php

namespace Rubix\ML\Backends\Tasks;

use Rubix\ML\Learner;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\Metric;

/**
 * Train and Validate
 *
 * A routine to train using a training set and subsequently cross-validate the model using a
 * testing set and scoring metric.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TrainAndValidate extends Task
{
    /**
     * Train the learner and then return its validation score.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Dataset $training
     * @param \Rubix\ML\Datasets\Labeled $testing
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @return float
     */
    public static function score(
        Learner $estimator,
        Dataset $training,
        Labeled $testing,
        Metric $metric
    ) : float {
        $estimator->train($training);

        $predictions = $estimator->predict($testing);

        $score = $metric->score($predictions, $testing->labels());

        return $score;
    }

    /**
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Dataset $training
     * @param \Rubix\ML\Datasets\Labeled $testing
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     */
    public function __construct(
        Learner $estimator,
        Dataset $training,
        Labeled $testing,
        Metric $metric
    ) {
        parent::__construct([self::class, 'score'], [
            $estimator, $training, $testing, $metric,
        ]);
    }
}

<?php

namespace Rubix\ML\Backends\Tasks;

use Rubix\ML\Learner;
use Rubix\ML\Datasets\Dataset;

class TrainLearner extends Task
{
    /**
     * Train a learner and return the instance.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Rubix\ML\Learner
     */
    public static function train(Learner $estimator, Dataset $dataset) : Learner
    {
        $estimator->train($dataset);

        return $estimator;
    }

    /**
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function __construct(Learner $estimator, Dataset $dataset)
    {
        parent::__construct([self::class, 'train'], [$estimator, $dataset]);
    }
}

<?php

namespace Rubix\ML\Benchmarks\Regressors;

use Rubix\ML\Regressors\KDNeighborsRegressor;
use Rubix\ML\Datasets\Generators\Hyperplane;

/**
 * @Groups({"Regressors"})
 */
class KDNeighborsRegressorBench
{
    protected const TRAINING_SIZE = 2500;

    protected const TESTING_SIZE = 10000;

    /**
     * @var \Rubix\ML\Datasets\Labeled;
     */
    public $training;

    /**
     * @var \Rubix\ML\Datasets\Labeled;
     */
    public $testing;

    /**
     * @var \Rubix\ML\Regressors\KDNeighborsRegressor
     */
    protected $estimator;

    public function setUpTrainPredict() : void
    {
        $generator = new Hyperplane([1, 5.5, -7, 0.01], 0.0);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->estimator = new KDNeighborsRegressor(5);
    }

    /**
     * @Iterations(3)
     * @BeforeMethods({"setUpTrainPredict"})
     * @OutputTimeUnit("seconds", precision=3)
     */
    public function bench_train_predict() : void
    {
        $this->estimator->train($this->training);

        $this->estimator->predict($this->testing);
    }
}

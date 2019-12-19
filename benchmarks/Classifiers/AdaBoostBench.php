<?php

namespace Rubix\ML\Benchmarks\Classifiers;

use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;

class AdaBoostBench
{
    protected const TRAINING_SIZE = 2500;

    protected const TESTING_SIZE = 5000;

    /**
     * @var \Rubix\ML\Datasets\Labeled;
     */
    public $training;

    /**
     * @var \Rubix\ML\Datasets\Labeled;
     */
    public $testing;

    /**
     * @var \Rubix\ML\Classifiers\AdaBoost
     */
    protected $estimator;

    public function setUpTrainPredict() : void
    {
        $generator = new Agglomerate([
            'red' => new Blob([255, 32, 0], 30.),
            'green' => new Blob([0, 128, 0], 10.),
            'blue' => new Blob([0, 32, 255], 20.),
        ], [2, 3, 4]);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->estimator = new AdaBoost();
    }

    /**
     * @BeforeMethods({"setUpTrainPredict"})
     */
    public function bench_train_predict() : void
    {
        $this->estimator->train($this->training);

        $this->estimator->predict($this->testing);
    }
}

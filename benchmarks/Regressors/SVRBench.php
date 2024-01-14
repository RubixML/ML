<?php

namespace Rubix\ML\Benchmarks\Regressors;

use Rubix\ML\Regressors\SVR;
use Rubix\ML\Datasets\Generators\Hyperplane;

/**
 * @Groups({"Regressors"})
 * @BeforeMethods({"setUp"})
 */
class SVRBench
{
    protected const TRAINING_SIZE = 10000;

    protected const TESTING_SIZE = 10000;

    /**
     * @var \Rubix\ML\Datasets\Labeled;
     */
    protected $training;

    /**
     * @var \Rubix\ML\Datasets\Labeled;
     */
    protected $testing;

    /**
     * @var SVR
     */
    protected $estimator;

    public function setUp() : void
    {
        $generator = new Hyperplane([1, 5.5, -7, 0.01], 0.0);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->estimator = new SVR();
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("seconds", precision=3)
     */
    public function trainPredict() : void
    {
        $this->estimator->train($this->training);

        $this->estimator->predict($this->testing);
    }
}

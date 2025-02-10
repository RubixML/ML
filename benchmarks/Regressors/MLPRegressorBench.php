<?php

namespace Rubix\ML\Benchmarks\Regressors;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\Regressors\MLPRegressor;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\Datasets\Generators\Hyperplane;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;

/**
 * @Groups({"Regressors"})
 * @BeforeMethods({"setUp"})
 */
class MLPRegressorBench
{
    protected const int TRAINING_SIZE = 10000;

    protected const int TESTING_SIZE = 10000;

    protected Labeled $training;

    protected Labeled $testing;

    protected MLPRegressor $estimator;

    public function setUp() : void
    {
        $generator = new Hyperplane([1, 5.5, -7, 0.01], 0.0);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->estimator = new MLPRegressor([
            new Dense(100),
            new Activation(new ReLU()),
        ]);
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

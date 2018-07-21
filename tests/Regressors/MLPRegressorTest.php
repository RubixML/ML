<?php

namespace Rubix\Tests\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\Regressor;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\Regressors\MLPRegressor;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class MLPRegressorTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = Labeled::restore(dirname(__DIR__) . '/mpg.dataset');

        $this->testing = $this->training->randomize()->head(3);

        $this->estimator = new MLPRegressor([
            new Dense(20, new ELU()),
        ], 5, new Adam(0.01), 1e-4, new MeanSquaredError(), 0.1, 3, 1e-4, 100);
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(MLPRegressor::class, $this->estimator);
        $this->assertInstanceOf(Regressor::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0], '', 3);
        $this->assertEquals($this->testing->label(1), $predictions[1], '', 3);
        $this->assertEquals($this->testing->label(2), $predictions[2], '', 3);
    }

    public function test_train_with_unlabeled()
    {
        $dataset = new Unlabeled([['bad']]);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train($dataset);
    }
}

<?php

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Estimator;
use Rubix\Engine\Estimators\Regressor;
use Rubix\Engine\Estimators\Persistable;
use Rubix\Engine\Estimators\DummyRegressor;
use Rubix\Engine\Transformers\Strategies\BlurryMean;
use PHPUnit\Framework\TestCase;

class DummyRegressorTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $samples = [
            [4, 91.0, 1795],
            [6, 225.0, 3651],
            [6, 250.0, 3574],
            [6, 250.0, 3645],
            [6, 258.0, 3193],
            [4, 97.0, 1825],
            [4, 85.0, 1990],
            [4, 87.0, 2155],
            [4, 130.0, 3150],
            [8, 318.0, 3940],
            [4, 120.0, 3270],
            [8, 260.0, 3365],
        ];

        $labels = [
            33.0, 20.0, 18.0, 18.5, 17.5, 29.5,
            32.0, 28.0, 20.0, 13.0, 19.0, 19.9,
        ];

        $this->training = new Supervised($samples, $labels);

        $this->testing = new Supervised([[6, 150.0, 2825]], [20]);

        $this->estimator = new DummyRegressor(new BlurryMean());
    }

    public function test_build_dummy_estimator()
    {
        $this->assertInstanceOf(DummyRegressor::class, $this->estimator);
        $this->assertInstanceOf(Regressor::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals(20, $predictions[0]->outcome(), '', 5);
    }
}

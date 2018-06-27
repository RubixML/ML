<?php

namespace Rubix\Tests\Regressors;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Regressor;
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\Transformers\Strategies\BlurryMean;
use PHPUnit\Framework\TestCase;

class DummyRegressorTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $samples = [
            [4, 91.0, 1795], [6, 225.0, 3651],
            [6, 250.0, 3574], [6, 250.0, 3645],
            [6, 258.0, 3193], [4, 97.0, 1825],
            [4, 85.0, 1990], [4, 87.0, 2155],
            [4, 130.0, 3150], [8, 318.0, 3940],
            [4, 120.0, 3270], [8, 260.0, 3365],
            [6, 231.0, 3245], [8, 360.0, 3940],
            [4, 89.00, 1925], [4, 141.0, 3230],
            [4, 107.0, 2205], [4, 144.0, 2665],
            [6, 168.0, 2910], [8, 350.0, 4165],
            [8, 400.0, 4220], [4, 97.00, 1985],
            [6, 232.0, 3265], [6, 163.0, 3140],
        ];

        $labels = [
            33.0, 20.0, 18.0, 18.5, 17.5, 29.5,
            32.0, 28.0, 20.0, 13.0, 19.0, 19.9,
            21.5, 18.5, 14.0, 28.1, 36.0, 32.0,
            32.7, 15.5, 16.0, 30.0, 20.2, 17.0,
        ];

        $this->training = new Labeled($samples, $labels);

        $this->testing = new Labeled([[6, 150.0, 2825]], [20]);

        $this->estimator = new DummyRegressor(new BlurryMean());
    }

    public function test_build_dummy_regressor()
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

        $this->assertEquals(23, $predictions[0], '', INF);
    }
}

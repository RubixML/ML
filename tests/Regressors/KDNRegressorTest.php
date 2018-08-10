<?php

namespace Rubix\Tests\Regressors;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\KDNRegressor;
use Rubix\ML\Kernels\Distance\Minkowski;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class KDNRegressorTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = Labeled::restore(dirname(__DIR__) . '/mpg.dataset');

        $this->testing = $this->training->randomize()->head(3);

        $this->estimator = new KDNRegressor(3, 10, new Minkowski(3.0));
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(KDNRegressor::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::REGRESSOR, $this->estimator->type());
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

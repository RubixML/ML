<?php

namespace Rubix\Tests\Regressors;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\Other\Strategies\BlurryMean;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class DummyRegressorTest extends TestCase
{
    const TOLERANCE = INF;

    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = Labeled::restore(dirname(__DIR__) . '/mpg.dataset');

        $this->testing = $this->training->randomize()->head(3);

        $this->estimator = new DummyRegressor(new BlurryMean());
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(DummyRegressor::class, $this->estimator);
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

        $this->assertEquals($this->testing->label(0), $predictions[0], '', self::TOLERANCE);
        $this->assertEquals($this->testing->label(1), $predictions[1], '', self::TOLERANCE);
        $this->assertEquals($this->testing->label(2), $predictions[2], '', self::TOLERANCE);
    }

    public function test_train_with_unlabeled()
    {
        $dataset = new Unlabeled([['bad']]);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train($dataset);
    }
}

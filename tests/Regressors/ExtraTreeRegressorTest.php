<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\CART;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\ExtraTreeRegressor;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class ExtraTreeRegressorTest extends TestCase
{
    const TOLERANCE = 3;

    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = unserialize(file_get_contents(dirname(__DIR__) . '/mpg.dataset') ?: '');

        $this->testing = $this->training->randomize()->head(3);

        $this->estimator = new ExtraTreeRegressor(30, 3, 0., null, 1e-4);
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(ExtraTreeRegressor::class, $this->estimator);
        $this->assertInstanceOf(CART::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::REGRESSOR, $this->estimator->type());
    }

    public function test_train_predict()
    {
        $this->estimator->train($this->training);

        foreach ($this->estimator->predict($this->testing) as $i => $prediction) {
            $this->assertEquals($this->testing->label($i), $prediction, '', self::TOLERANCE);
        }
    }

    public function test_train_with_unlabeled()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick());
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Graph\KDTree;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\KDNRegressor;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\ML\Datasets\Generators\HalfMoon;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class KDNRegressorTest extends TestCase
{
    const TRAIN_SIZE = 300;
    const TEST_SIZE = 5;
    const TOLERANCE = 10;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new HalfMoon(4., -7., 1., 90, 0.02);

        $this->estimator = new KDNRegressor(3, 10, new Minkowski(3.0), true);
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(KDNRegressor::class, $this->estimator);
        $this->assertInstanceOf(KDTree::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::REGRESSOR, $this->estimator->type());
    }

    public function test_train_predict()
    {
        $testing = $this->generator->generate(self::TEST_SIZE);
        
        $this->estimator->train($this->generator->generate(self::TRAIN_SIZE));

        $predictions = $this->estimator->predict($testing);

        foreach ($predictions as $i => $prediction) {
            $this->assertEquals($testing->label($i), $prediction, '', self::TOLERANCE);
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

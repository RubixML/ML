<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Graph\BallTree;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Regressors\RadiusNeighborsRegressor;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class RadiusNeighborsRegressorTest extends TestCase
{
    protected const TEST_SIZE = 5;
    protected const MIN_SCORE = 0.8;

    protected $training;

    protected $testing;

    protected $estimator;

    protected $metric;

    public function setUp()
    {
        $this->training = unserialize(file_get_contents(dirname(__DIR__) . '/mpg.dataset') ?: '');

        $this->testing = $this->training->randomize()->head(self::TEST_SIZE);

        $this->estimator = new RadiusNeighborsRegressor(1.5, new Euclidean(), true, 20);
        
        $this->metric = new RSquared();
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(RadiusNeighborsRegressor::class, $this->estimator);
        $this->assertInstanceOf(BallTree::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);

        $this->assertEquals(Estimator::REGRESSOR, $this->estimator->type());

        $this->assertNotContains(DataType::CATEGORICAL, $this->estimator->compatibility());
        $this->assertContains(DataType::CONTINUOUS, $this->estimator->compatibility());

        $this->assertFalse($this->estimator->trained());

        $this->assertEquals(0, $this->estimator->height());
    }

    public function test_train_predict()
    {
        $this->estimator->train($this->training);

        $this->assertTrue($this->estimator->trained());

        $this->assertGreaterThan(0, $this->estimator->height());

        $predictions = $this->estimator->predict($this->testing);

        $score = $this->metric->score($predictions, $this->testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function test_train_with_unlabeled()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick());
    }

    public function test_train_incompatible()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick([['bad']]));
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

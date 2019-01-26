<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\KNNRegressor;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\ML\Datasets\Generators\HalfMoon;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class KNNRegressorTest extends TestCase
{
    const TRAIN_SIZE = 300;
    const TEST_SIZE = 10;
    const MIN_SCORE = 0.9;

    protected $generator;

    protected $estimator;
    
    protected $metric;

    public function setUp()
    {
        $this->generator = new HalfMoon(4., -7., 1., 90, 0.02);

        $this->estimator = new KNNRegressor(3, new Minkowski(3.0), true);

        $this->metric = new RSquared();
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(KNNRegressor::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::REGRESSOR, $this->estimator->type());
    }
    
    public function test_train_partial_predict_proba()
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $testing = $this->generator->generate(self::TEST_SIZE);

        $folds = $training->fold(3);
        
        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

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

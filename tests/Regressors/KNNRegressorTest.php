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
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class KNNRegressorTest extends TestCase
{
    const TRAIN_SIZE = 300;
    const TEST_SIZE = 5;
    const TOLERANCE = 5;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new HalfMoon(4., -7., 1., 90, 0.02);

        $this->estimator = new KNNRegressor(3, new Minkowski(3.0));
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
        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $transformer = new ZScaleStandardizer();

        $transformer->fit($dataset);
        $dataset->apply($transformer);

        $testing = $dataset->randomize()->take(self::TEST_SIZE);

        $folds = $dataset->fold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        foreach ($this->estimator->predict($testing) as $i => $prediction) {
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

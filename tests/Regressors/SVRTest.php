<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Regressors\SVR;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\DataType;
use Rubix\ML\Kernels\SVM\Polynomial;
use Rubix\ML\Datasets\Generators\HalfMoon;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class SVRTest extends TestCase
{
    protected const TRAIN_SIZE = 300;
    protected const TEST_SIZE = 10;
    protected const MIN_SCORE = -INF;

    protected const RANDOM_SEED = 0;

    protected $generator;

    protected $estimator;
    
    protected $metric;

    public function setUp()
    {
        $this->generator = new HalfMoon(0., 0., 1., 90, 0.02);

        $this->estimator = new SVR(0.1, 1e-3, new Polynomial(4), false, 1e-3);

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(SVR::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);

        $this->assertSame(Estimator::REGRESSOR, $this->estimator->type());

        $this->assertNotContains(DataType::CATEGORICAL, $this->estimator->compatibility());
        $this->assertContains(DataType::CONTINUOUS, $this->estimator->compatibility());

        $this->assertFalse($this->estimator->trained());
    }

    public function test_train_predict()
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $testing = $dataset->randomize()->take(self::TEST_SIZE);

        $this->estimator->train($dataset);

        $this->assertTrue($this->estimator->trained());

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

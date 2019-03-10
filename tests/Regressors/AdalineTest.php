<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Regressors\Adaline;
use Rubix\ML\Other\Loggers\BlackHole;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class AdalineTest extends TestCase
{
    protected const TEST_SIZE = 5;
    protected const MIN_SCORE = 0.6;

    protected const RANDOM_SEED = 0;

    protected $estimator;

    protected $training;

    protected $testing;

    protected $metric;

    public function setUp()
    {
        $this->training = unserialize(file_get_contents(dirname(__DIR__) . '/mpg.dataset') ?: '');

        $this->testing = $this->training->randomize()->take(self::TEST_SIZE);

        $this->estimator = new Adaline(1, new Adam(0.01), 1e-4, 100, 1e-3, new HuberLoss(1.));

        $this->metric = new RSquared();

        $this->estimator->setLogger(new BlackHole());

        srand(self::RANDOM_SEED);
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(Adaline::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);

        $this->assertEquals(Estimator::REGRESSOR, $this->estimator->type());

        $this->assertNotContains(DataType::CATEGORICAL, $this->estimator->compatibility());
        $this->assertContains(DataType::CONTINUOUS, $this->estimator->compatibility());

        $this->assertFalse($this->estimator->trained());
    }

    public function test_train_predict()
    {
        $this->estimator->train($this->training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($this->testing);

        $score = $this->metric->score($predictions, $this->testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function test_train_with_unlabeled()
    {
        $dataset = new Unlabeled([['bad']]);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train($dataset);
    }

    public function test_train_incompatible()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick([['bad']]));
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict($this->testing);
    }
}

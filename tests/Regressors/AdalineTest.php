<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\RanksFeatures;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\Adaline;
use Rubix\ML\Other\Loggers\BlackHole;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Datasets\Generators\Hyperplane;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Regressors
 * @covers \Rubix\ML\Regressors\Adaline
 */
class AdalineTest extends TestCase
{
    /**
     * The number of samples in the training set.
     *
     * @var int
     */
    protected const TRAIN_SIZE = 300;

    /**
     * The number of samples in the validation set.
     *
     * @var int
     */
    protected const TEST_SIZE = 20;

    /**
     * The minimum validation score required to pass the test.
     *
     * @var float
     */
    protected const MIN_SCORE = 0.9;

    /**
     * Constant used to see the random number generator.
     *
     * @var int
     */
    protected const RANDOM_SEED = 0;

    /**
     * @var \Rubix\ML\Datasets\Generators\Hyperplane
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Regressors\Adaline
     */
    protected $estimator;

    /**
     * @var \Rubix\ML\CrossValidation\Metrics\RSquared
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Hyperplane([1, 5.5, -7, 0.01], 0.0);

        $this->estimator = new Adaline(1, new Adam(0.01), 1e-4, 100, 1e-3, 5, new HuberLoss(1.0));

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Adaline::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(RanksFeatures::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function badBatchSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Adaline(-100);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    /**
     * @test
     */
    public function params() : void
    {
        $expected = [
            'batch_size' => 1,
            'optimizer' => new Adam(0.01),
            'alpha' => 1e-4,
            'epochs' => 100,
            'min_change' => 1e-3,
            'window' => 5,
            'cost_fn' => new HuberLoss(1.0),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @test
     */
    public function trainPredictImportances() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $losses = $this->estimator->steps();

        $this->assertIsArray($losses);
        $this->assertContainsOnly('float', $losses);

        $importances = $this->estimator->featureImportances();

        $this->assertIsArray($importances);
        $this->assertCount(4, $importances);
        $this->assertContainsOnly('float', $importances);
        $this->assertEqualsWithDelta(1.0, array_sum($importances), 1e-8);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    /**
     * @test
     */
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick([['bad']], [2]));
    }

    /**
     * @test
     */
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }
}

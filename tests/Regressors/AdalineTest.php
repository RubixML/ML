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
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\Adaline;
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
    protected const TRAIN_SIZE = 512;

    /**
     * The number of samples in the validation set.
     *
     * @var int
     */
    protected const TEST_SIZE = 256;

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
     * @var Hyperplane
     */
    protected $generator;

    /**
     * @var Adaline
     */
    protected $estimator;

    /**
     * @var RSquared
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Hyperplane([1.0, 5.5, -7, 0.01], 0.0, 1.0);

        $this->estimator = new Adaline(32, new Adam(0.001), 1e-4, 100, 1e-4, 5, new HuberLoss(1.0));

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
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
            'batch size' => 32,
            'optimizer' => new Adam(0.001),
            'l2 penalty' => 1e-4,
            'epochs' => 100,
            'min change' => 1e-4,
            'window' => 5,
            'cost fn' => new HuberLoss(1.0),
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

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnly('float', $losses);

        $importances = $this->estimator->featureImportances();

        $this->assertIsArray($importances);
        $this->assertCount(4, $importances);
        $this->assertContainsOnly('float', $importances);

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
}

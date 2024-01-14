<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Encoding;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Graphviz;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\Regressors\MLPRegressor;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\Datasets\Generators\SwissRoll;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Regressors
 * @covers \Rubix\ML\Regressors\MLPRegressor
 */
class MLPRegressorTest extends TestCase
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
     * @var SwissRoll
     */
    protected $generator;

    /**
     * @var MLPRegressor
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
        $this->generator = new SwissRoll(4.0, -7.0, 0.0, 1.0, 21.0, 0.5);

        $this->estimator = new MLPRegressor([
            new Dense(32),
            new Activation(new SiLU()),
            new Dense(16),
            new Activation(new SiLU()),
            new Dense(8),
            new Activation(new SiLU()),
        ], 32, new Adam(0.01), 1e-4, 100, 1e-4, 5, 0.1, new LeastSquares(), new RMSE());

        $this->metric = new RSquared();

        $this->estimator->setLogger(new BlackHole());

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
        $this->assertInstanceOf(MLPRegressor::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
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

        new MLPRegressor([], -100);
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
            'hidden layers' => [
                new Dense(32),
                new Activation(new SiLU()),
                new Dense(16),
                new Activation(new SiLU()),
                new Dense(8),
                new Activation(new SiLU()),
            ],
            'batch size' => 32,
            'optimizer' => new Adam(0.01),
            'l2 penalty' => 1e-4,
            'epochs' => 100,
            'min change' => 1e-4,
            'window' => 5,
            'hold out' => 0.1,
            'cost fn' => new LeastSquares(),
            'metric' => new RMSE(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @test
     */
    public function trainPartialPredict() : void
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $testing = $dataset->randomize()->take(self::TEST_SIZE);

        $folds = $dataset->fold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        $this->assertTrue($this->estimator->trained());

        $dot = $this->estimator->exportGraphviz();

        // Graphviz::dotToImage($dot)->saveTo(new Filesystem('test.png'));

        $this->assertInstanceOf(Encoding::class, $dot);
        $this->assertStringStartsWith('digraph Tree {', $dot);

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnly('float', $losses);

        $scores = $this->estimator->scores();

        $this->assertIsArray($scores);
        $this->assertContainsOnly('float', $scores);

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

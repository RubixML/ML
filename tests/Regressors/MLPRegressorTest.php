<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Regressors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
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

#[Group('Regressors')]
#[CoversClass(MLPRegressor::class)]
class MLPRegressorTest extends TestCase
{
    /**
     * The number of samples in the training set.
     */
    protected const int TRAIN_SIZE = 512;

    /**
     * The number of samples in the validation set.
     */
    protected const int TEST_SIZE = 256;

    /**
     * The minimum validation score required to pass the test.
     */
    protected const float MIN_SCORE = 0.9;

    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected SwissRoll $generator;

    protected MLPRegressor $estimator;

    protected RSquared $metric;

    protected function setUp() : void
    {
        $this->generator = new SwissRoll(x: 4.0, y: -7.0, z: 0.0, scale: 1.0, depth: 21.0, noise: 0.5);

        $this->estimator = new MLPRegressor(
            hiddenLayers: [
                new Dense(32),
                new Activation(new SiLU()),
                new Dense(16),
                new Activation(new SiLU()),
                new Dense(8),
                new Activation(new SiLU()),
            ],
            batchSize: 32,
            optimizer: new Adam(0.01),
            epochs: 100,
            minChange: 1e-4,
            evalInterval: 3,
            window: 5,
            holdOut: 0.1,
            costFn: new LeastSquares(),
            metric: new RMSE()
        );

        $this->metric = new RSquared();

        $this->estimator->setLogger(new BlackHole());

        srand(self::RANDOM_SEED);
    }

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    public function testBadBatchSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new MLPRegressor(hiddenLayers: [], batchSize: -100);
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    public function testCompatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    public function testParams() : void
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
            'epochs' => 100,
            'min change' => 1e-4,
            'eval interval' => 3,
            'window' => 5,
            'hold out' => 0.1,
            'cost fn' => new LeastSquares(),
            'metric' => new RMSE(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    public function testTrainPartialPredict() : void
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

        $this->assertStringStartsWith('digraph Tree {', (string) $dot);

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnlyFloat($losses);

        $scores = $this->estimator->scores();

        $this->assertIsArray($scores);
        $this->assertContainsOnlyFloat($scores);

        $predictions = $this->estimator->predict($testing);

        /** @var list<int|float> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function testTrainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: [2]));
    }

    public function testPredictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

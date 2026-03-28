<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Regressors\MLPRegressors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\NeuralNet\Layers\Dense\Dense;
use Rubix\ML\Regressors\MLPRegressor\MLPRegressor;
use Rubix\ML\NeuralNet\Optimizers\Adam\Adam;
use Rubix\ML\NeuralNet\Layers\Activation\Activation;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\Datasets\Generators\SwissRoll\SwissRoll;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU\SiLU;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares\LeastSquares;
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
            metric: new RMSE(),
            packSamples: true,
        );

        $this->metric = new RSquared();

        $this->estimator->setLogger(new BlackHole());

        srand(self::RANDOM_SEED);
    }

    #[Test]
    #[TestDox('Assert pre conditions')]
    public function testAssertPreConditions() : void
    {
        self::assertFalse($this->estimator->trained());
    }

    #[Test]
    #[TestDox('Bad batch size')]
    public function testBadBatchSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new MLPRegressor(hiddenLayers: [], batchSize: -100);
    }

    #[Test]
    #[TestDox('Type')]
    public function testType() : void
    {
        self::assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    #[Test]
    #[TestDox('Compatibility')]
    public function testCompatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        self::assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    #[TestDox('Params')]
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

        self::assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    #[TestDox('Train partial predict')]
    public function testTrainPartialPredict() : void
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $testing = $dataset->randomize()->take(self::TEST_SIZE);

        $folds = $dataset->fold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        self::assertTrue($this->estimator->trained());

        $dot = $this->estimator->exportGraphviz();

        // Graphviz::dotToImage($dot)->saveTo(new Filesystem('test.png'));

        self::assertStringStartsWith('digraph Tree {', (string) $dot);

        $losses = $this->estimator->losses();

        self::assertIsArray($losses);
        self::assertContainsOnlyFloat($losses);

        $scores = $this->estimator->scores();

        self::assertIsArray($scores);
        self::assertContainsOnlyFloat($scores);

        $predictions = $this->estimator->predict($testing);

        /** @var list<int|float> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        self::assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    #[TestDox('Predict count matches number of samples')]
    public function testPredictCountMatchesNumberOfSamples() : void
    {
        [$testing] = $this->trainEstimatorAndGetTestingSet();

        $predictions = $this->estimator->predict($testing);

        self::assertCount($testing->numSamples(), $predictions);
    }

    #[Test]
    #[TestDox('Predict returns numeric finite values')]
    public function testPredictReturnsNumericFiniteValues() : void
    {
        [$testing] = $this->trainEstimatorAndGetTestingSet();

        $predictions = $this->estimator->predict($testing);

        self::assertCount($testing->numSamples(), $predictions);

        foreach ($predictions as $prediction) {
            self::assertIsNumeric($prediction);
            self::assertFalse(is_nan((float) $prediction));
            self::assertTrue(is_finite((float) $prediction));
        }
    }

    #[Test]
    #[TestDox('Predict is repeatable for same model and dataset')]
    public function testPredictIsRepeatableForSameModelAndDataset() : void
    {
        [$testing] = $this->trainEstimatorAndGetTestingSet();

        $predictions1 = $this->estimator->predict($testing);
        $predictions2 = $this->estimator->predict($testing);

        self::assertCount($testing->numSamples(), $predictions1);
        self::assertCount($testing->numSamples(), $predictions2);

        foreach ($predictions1 as $i => $prediction) {
            self::assertEqualsWithDelta((float) $prediction, (float) $predictions2[$i], 1e-12);
        }
    }

    #[Test]
    #[TestDox('Predict does not mutate dataset samples or labels')]
    public function testPredictDoesNotMutateDataset() : void
    {
        [$testing] = $this->trainEstimatorAndGetTestingSet();

        $samplesBefore = $testing->samples();
        $labelsBefore = $testing->labels();

        $predictions = $this->estimator->predict($testing);

        self::assertCount($testing->numSamples(), $predictions);
        self::assertEquals($samplesBefore, $testing->samples());
        self::assertEquals($labelsBefore, $testing->labels());
    }

    #[Test]
    #[TestDox('Serialization preserves predict output')]
    public function testSerializationPreservesPredictOutput() : void
    {
        [$testing] = $this->trainEstimatorAndGetTestingSet();

        $predictionsBefore = $this->estimator->predict($testing);

        $copy = unserialize(serialize($this->estimator));

        self::assertInstanceOf(MLPRegressor::class, $copy);
        self::assertTrue($copy->trained());

        $predictionsAfter = $copy->predict($testing);

        self::assertCount($testing->numSamples(), $predictionsAfter);

        foreach ($predictionsAfter as $i => $prediction) {
            self::assertEqualsWithDelta((float) $predictionsBefore[$i], (float) $prediction, 1e-8);
        }
    }

    #[Test]
    #[TestDox('Train incompatible')]
    public function testTrainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: [2]));
    }

    #[Test]
    #[TestDox('Predict untrained')]
    public function testPredictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    /**
     * @return array{0: Unlabeled}
     */
    private function trainEstimatorAndGetTestingSet() : array
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $testing = $dataset->randomize()->take(self::TEST_SIZE);

        $folds = $dataset->fold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        return [$testing];
    }
}

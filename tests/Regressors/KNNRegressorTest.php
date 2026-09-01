<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Regressors;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RunInSeparateProcess;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\KNNRegressor;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\ML\Datasets\Generators\HalfMoon;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Swoole;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use PHPUnit\Framework\TestCase;

#[Group('Regressors')]
#[CoversClass(KNNRegressor::class)]
class KNNRegressorTest extends TestCase
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

    protected HalfMoon $generator;

    protected KNNRegressor $estimator;

    protected RSquared $metric;

    protected ?Backend $backend = null;

    public static function trainedStateCases() : Generator
    {
        yield 'three-fold partial fit' => [self::TRAIN_SIZE, 3];
    }

    /**
     * @return Generator<string, array{backend: Backend}>
     */
    public static function provideBackends() : Generator
    {
        $serialBackend = new Serial();

        yield (string) $serialBackend => [
            'backend' => $serialBackend,
        ];

        $ampBackend = new Amp();

        yield (string) $ampBackend => [
            'backend' => $ampBackend,
        ];

        if (ExtensionIsLoaded::with('swoole')->passes()) {
            $swooleBackend = new Swoole();

            yield (string) $swooleBackend => [
                'backend' => $swooleBackend,
            ];
        }
    }

    protected function setUp() : void
    {
        $this->generator = new HalfMoon(x: 4.0, y: -7.0, scale: 1.0, rotation: 90, noise: 0.25);

        $this->estimator = new KNNRegressor(k: 10, weighted: true, kernel:  new Minkowski(3.0));

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    protected function tearDown() : void
    {
        $this->backend?->shutdown();
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    #[Test]
    public function badK() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new KNNRegressor(k: 0);
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    #[Test]
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $expected = [
            'k' => 10,
            'weighted' => true,
            'kernel' => new Minkowski(3.0),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[DataProvider('provideBackends')]
    #[Test]
    #[RunInSeparateProcess]
    public function trainPredict(Backend $backend) : void
    {
        $this->backend = $backend;

        $this->estimator->setBackend($backend);

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        /** @var list<int|float> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    #[RunInSeparateProcess]
    public function predictionsAgreeAcrossBackends() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $serial = new KNNRegressor(k: 10, weighted: true, kernel: new Minkowski(3.0));

        $serial->train($training);

        $serialPredictions = $serial->predict($testing);

        $ampBackend = new Amp();

        $this->backend = $ampBackend;

        $amp = new KNNRegressor(k: 10, weighted: true, kernel: new Minkowski(3.0));

        $amp->setBackend($ampBackend);

        $amp->train($training);

        $ampPredictions = $amp->predict($testing);

        $this->assertEquals($serialPredictions, $ampPredictions);
    }

    #[Test]
    public function trainPartialPredict() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $folds = $training->fold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        /** @var list<int|float> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function weightedPredictionAlignsLabelsAndWeights() : void
    {
        // Samples in a different input order than their ranking by proximity,
        // so a distance-sorted weight table must be re-aligned against labels.
        $this->estimator = new KNNRegressor(3, true);

        $this->estimator->train(Labeled::quick(
            [[1.0], [2.0], [3.0]],
            [0.0, 10.0, 30.0]
        ));

        $predictions = $this->estimator->predict(Unlabeled::quick([[2.0]]));

        $this->assertEqualsWithDelta([12.5], $predictions, 1e-8);
    }

    #[Test]
    public function weightedPredictionAlignsLabelsAndWeightsAtBoundary() : void
    {
        $this->estimator = new KNNRegressor(3, true);

        $this->estimator->train(Labeled::quick(
            [[1.0], [5.0], [10.0]],
            [0.0, 5.0, 100.0]
        ));

        $predictions = $this->estimator->predict(Unlabeled::quick([[10.0]]));

        $this->assertEqualsWithDelta(79.605263158, $predictions[0], 1e-8);
    }

    #[Test]
    public function weightedPredictionWithKLimit() : void
    {
        $this->estimator = new KNNRegressor(2, true);

        $this->estimator->train(Labeled::quick(
            [[1.0], [0.5], [3.0]],
            [1.0, 2.0, 100.0]
        ));

        $predictions = $this->estimator->predict(Unlabeled::quick([[0.0]]));

        // Only the two nearest neighbors (labels 1 and 2) contribute; the
        // far outlier (label 100) must be excluded by the k limit.
        $expected = (1.0 * (1.0 / (1.0 + 1.0)) + 2.0 * (1.0 / (1.0 + 0.5)))
            / ((1.0 / (1.0 + 1.0)) + (1.0 / (1.0 + 0.5)));

        $this->assertEqualsWithDelta($expected, $predictions[0], 1e-8);
    }

    #[Test]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: [2]));
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    #[DataProvider('trainedStateCases')]
    #[Test]
    public function becomesTrainedAfterPartialFitting(int $trainSize, int $folds) : void
    {
        $training = $this->generator->generate($trainSize);

        $parts = $training->fold($folds);

        $this->estimator->train($parts[0]);

        for ($i = 1; $i < $folds; ++$i) {
            $this->estimator->partial($parts[$i]);
        }

        $this->assertTrue($this->estimator->trained());
    }

    #[Test]
    #[TestDox('Backend is transient and resolved lazily')]
    public function backendIsTransient() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->setBackend(new Serial());

        $this->estimator->train($training);

        self::assertTrue($this->estimator->trained());

        self::assertArrayNotHasKey('backend', $this->estimator->__serialize());

        $copy = unserialize(serialize($this->estimator));

        self::assertInstanceOf(KNNRegressor::class, $copy);
        self::assertTrue($copy->trained());

        $predictions = $copy->predict($training);

        self::assertCount(self::TRAIN_SIZE, $predictions);

        self::assertArrayNotHasKey('backend', $copy->__serialize());
    }
}

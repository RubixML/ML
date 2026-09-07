<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\AnomalyDetectors;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RunInSeparateProcess;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\AnomalyDetectors\IsolationForest;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Swoole;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use PHPUnit\Framework\TestCase;

#[Group('AnomalyDetectors')]
#[CoversClass(IsolationForest::class)]
class IsolationForestTest extends TestCase
{
    /**
     * The number of samples in the training set.
     */
    protected const int TRAIN_SIZE = 2000;

    /**
     * The number of samples in the validation set.
     */
    protected const int TEST_SIZE = 1000;

    /**
     * The minimum validation score required to pass the test.
     */
    protected const float MIN_SCORE = 0.8;

    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected IsolationForest $estimator;

    protected FBeta $metric;

    protected ?Backend $backend = null;

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
        $this->generator = new Agglomerate(
            generators: [
                new Blob(
                    center: [0.0, 0.0],
                    stdDev: 2.0
                ),
                new Circle(
                    x: 0.0,
                    y: 0.0,
                    scale: 8.0,
                    noise: 1.0
                ),
            ],
            weights: [0.9, 0.1]
        );

        $this->estimator = new IsolationForest(
            estimators: 300,
            ratio: 0.2,
            contamination: 0.1
        );

        $this->metric = new FBeta();

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
    public function badNumEstimators() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new IsolationForest(estimators: -100);
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(EstimatorType::anomalyDetector(), $this->estimator->type());
    }

    #[Test]
    public function compatibility() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $expected = [
            'estimators' => 300,
            'ratio' => 0.2,
            'contamination' => 0.1,
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

        /** @var list<int|string> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
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

        self::assertInstanceOf(IsolationForest::class, $copy);
        self::assertTrue($copy->trained());

        $predictions = $copy->predict($training);

        self::assertCount(self::TRAIN_SIZE, $predictions);

        self::assertArrayNotHasKey('backend', $copy->__serialize());
    }

    #[Test]
    public function score() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->setBackend(new Serial());

        $this->estimator->train($training);

        $scores = $this->estimator->score($testing);

        $this->assertCount(self::TEST_SIZE, $scores);
        $this->assertContainsOnlyFloat($scores);

        foreach ($scores as $score) {
            $this->assertIsFloat($score);
            $this->assertFalse(is_nan($score));
            $this->assertTrue(is_finite($score));
            $this->assertGreaterThanOrEqual(0.0, $score);
            $this->assertLessThanOrEqual(1.0, $score);
        }
    }
}

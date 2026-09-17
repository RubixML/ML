<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Base;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RunInSeparateProcess;
use Rubix\ML\DataType;
use Rubix\ML\GridSearch;
use Rubix\ML\EstimatorType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Swoole;
use Rubix\ML\Specifications\ExtensionIsLoaded;

#[Group('MetaEstimators')]
#[CoversClass(GridSearch::class)]
class GridSearchTest extends TestCase
{
    protected const int TRAIN_SIZE = 512;

    protected const int TEST_SIZE = 256;

    protected const float MIN_SCORE = 0.9;

    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected GridSearch $estimator;

    protected Accuracy $metric;

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
                'inner' => new Circle(x: 0.0, y: 0.0, scale: 1.0, noise: 0.5),
                'middle' => new Circle(x: 0.0, y: 0.0, scale: 5.0, noise: 1.0),
                'outer' => new Circle(x: 0.0, y: 0.0, scale: 10.0, noise: 2.0),
            ]
        );

        $this->estimator = new GridSearch(
            class: KNearestNeighbors::class,
            params: [
                [1, 5, 10],
                [true],
                [
                    new Euclidean(),
                    new Manhattan(),
                ],
            ],
            metric: new FBeta(),
            validator: new HoldOut(0.2)
        );

        $this->metric = new Accuracy();

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
    public function type() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    #[Test]
    public function compatibility() : void
    {
        $this->assertEquals(DataType::all(), $this->estimator->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $expected = [
            'class' => KNearestNeighbors::class,
            'params' => [
                [1, 5, 10], [true], [new Euclidean(), new Manhattan()],
            ],
            'metric' => new FBeta(),
            'validator' => new HoldOut(0.2),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @param Backend $backend
     */
    #[DataProvider('provideBackends')]
    #[Test]
    #[RunInSeparateProcess]
    public function trainPredictBest(Backend $backend) : void
    {
        $this->backend = $backend;

        $this->estimator->setLogger(new BlackHole());
        $this->estimator->setBackend($backend);

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        /** @var list<int|string> $predictions */
        $predictions = $this->estimator->predict($testing);

        /** @var list<int|string> $labels */
        $labels = $testing->labels();

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);

        $expectedBest = [
            'k' => 10,
            'weighted' => true,
            'kernel' => new Manhattan(),
        ];

        $this->assertEquals($expectedBest, $this->estimator->base()->params());
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

        self::assertInstanceOf(GridSearch::class, $copy);
        self::assertTrue($copy->trained());

        $predictions = $copy->predict($training);

        self::assertCount($training->numSamples(), $predictions);

        self::assertArrayNotHasKey('backend', $copy->__serialize());
    }

    #[Test]
    public function resultsAreTableOfCombinationsAndScores() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->assertNotEmpty($this->estimator->scores());

        $progress = $this->estimator->results();

        $this->assertInstanceOf(Generator::class, $progress);

        $rows = iterator_to_array($progress);

        $this->assertCount(6, $rows);

        $metric = new FBeta();

        $expectedBest = [
            'k' => '10',
            'weighted' => 'true',
            'kernel' => 'Manhattan',
        ];

        $first = $rows[0];

        foreach ($expectedBest as $key => $value) {
            $this->assertArrayHasKey($key, $first);
            $this->assertSame($value, $first[$key]);
        }

        $this->assertArrayHasKey("{$metric}", $first);

        $scores = [];

        foreach ($rows as $row) {
            $this->assertSame(
                ['k', 'weighted', 'kernel', "{$metric}"],
                array_keys($row)
            );

            $scores[] = (float) $row["{$metric}"];
        }

        $sorted = $scores;

        rsort($sorted);

        $this->assertSame($sorted, $scores);
    }

    #[Test]
    public function fromNamedParams() : void
    {
        $estimator = GridSearch::fromNamedParams(
            KNearestNeighbors::class,
            params: [
                'kernel' => [new Manhattan(), new Euclidean()],
                'weighted' => [true],
                'k' => [1, 5, 10],
            ],
            metric: new FBeta(),
            validator: new HoldOut(0.2)
        );

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());

        $expectedBest = [
            'k' => 10,
            'weighted' => true,
            'kernel' => new Manhattan(),
        ];

        $this->assertEquals($expectedBest, $estimator->base()->params());

        $rows = iterator_to_array($estimator->results());

        $expectedFirst = [
            'k' => '10',
            'weighted' => 'true',
            'kernel' => 'Manhattan',
        ];

        foreach ($expectedFirst as $key => $value) {
            $this->assertSame($value, $rows[0][$key]);
        }

        $this->assertSame(
            ['k', 'weighted', 'kernel'],
            array_slice(array_keys($rows[0]), 0, 3)
        );
    }

    #[Test]
    public function fromNamedParamsFillsDefaults() : void
    {
        $estimator = GridSearch::fromNamedParams(
            KNearestNeighbors::class,
            params: [
                'k' => [1, 5, 10],
                'kernel' => [new Euclidean(), new Manhattan()],
            ]
        );

        $expected = [
            'class' => KNearestNeighbors::class,
            'params' => [
                [1, 5, 10],
                [false],
                [new Euclidean(), new Manhattan()],
            ],
            'metric' => new FBeta(),
            'validator' => new KFold(3),
        ];

        $this->assertEquals($expected, $estimator->params());
    }

    #[Test]
    public function fromNamedParamsRejectsUnknownParam() : void
    {
        $this->expectException(InvalidArgumentException::class);

        GridSearch::fromNamedParams(KNearestNeighbors::class, ['nope' => [true]]);
    }
}

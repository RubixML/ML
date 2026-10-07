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
use Rubix\ML\Report;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\FBeta;
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

        $rows = $this->estimator->results();

        $this->assertInstanceOf(Report::class, $rows);

        $this->assertCount(6, $rows);

        $metric = new FBeta();

        $scores = $this->estimator->scores();

        foreach ($this->estimator->combinations() as $i => $combination) {
            $row = $rows['Trial ' . ($i + 1)];

            $this->assertSame(
                ['k', 'weighted', 'kernel', "{$metric}"],
                array_keys($row)
            );

            $expectedParams = array_map(
                [Params::class, 'toString'],
                $combination
            );

            $this->assertSame($expectedParams, array_values(array_slice($row, 0, 3)));

            $this->assertSame((string) $scores[$i], $row["{$metric}"]);
        }
    }

    #[Test]
    public function bestIsNullsBeforeTraining() : void
    {
        $this->assertSame([null, null], $this->estimator->best());
    }

    #[Test]
    public function bestReturnsTopPerformingParamsAndScore() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        [$bestParams, $bestScore] = $this->estimator->best();

        $expectedParams = [
            'k' => '5',
            'weighted' => 'true',
            'kernel' => 'Euclidean',
        ];

        $this->assertSame($expectedParams, $bestParams);

        $this->assertSame(
            ['k', 'weighted', 'kernel'],
            array_keys($bestParams)
        );

        $this->assertSame(
            Params::toString(max($this->estimator->scores())),
            $bestScore
        );
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
            'k' => 5,
            'weighted' => true,
            'kernel' => new Euclidean(),
        ];

        $this->assertEquals($expectedBest, $estimator->base()->params());

        [$bestParams, $bestScore] = $estimator->best();

        $expectedBestParams = [
            'k' => '5',
            'weighted' => 'true',
            'kernel' => 'Euclidean',
        ];

        $this->assertSame($expectedBestParams, $bestParams);

        $this->assertSame(
            Params::toString(max($estimator->scores())),
            $bestScore
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
            'validator' => new KFold(5),
        ];

        $this->assertEquals($expected, $estimator->params());
    }

    #[Test]
    public function fillsEmptyTuplesWithDefaults() : void
    {
        $estimator = new GridSearch(
            class: KNearestNeighbors::class,
            params: [[], [], []],
        );

        $expected = [
            'class' => KNearestNeighbors::class,
            'params' => [
                [5],
                [false],
                [null],
            ],
            'metric' => new FBeta(),
            'validator' => new KFold(5),
        ];

        $this->assertEquals($expected, $estimator->params());
    }

    #[Test]
    public function fromNamedParamsRejectsUnknownParam() : void
    {
        $this->expectException(InvalidArgumentException::class);

        GridSearch::fromNamedParams(KNearestNeighbors::class, ['nope' => [true]]);
    }

    #[Test]
    public function rejectsScalarTuple() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GridSearch(KNearestNeighbors::class, [10, [1, 5]]);
    }

    #[Test]
    public function rejectsUnorderedParams() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GridSearch(KNearestNeighbors::class, ['k' => [1, 5]]);
    }

    #[Test]
    #[TestDox('Setup callback is invoked on each estimator before cross-validation')]
    public function setupIsCalledForEachEstimator() : void
    {
        $callCount = 0;
        $types = [];

        $this->estimator->setup(function (KNearestNeighbors $estimator) use (&$callCount, &$types) {
            ++$callCount;
            $types[] = $estimator::class;
        });

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        // 6 param combinations + 1 final best estimator
        $this->assertSame(7, $callCount);

        foreach ($types as $type) {
            $this->assertSame(KNearestNeighbors::class, $type);
        }
    }

    #[Test]
    #[TestDox('Setup callback returns $this for fluent chaining')]
    public function setupIsFluent() : void
    {
        $result = $this->estimator->setup(function (KNearestNeighbors $e) : void {
        });

        $this->assertSame($this->estimator, $result);
    }

    #[Test]
    #[TestDox('Setup closure is transient and excluded from serialization')]
    public function setupIsTransient() : void
    {
        $this->estimator->setup(function (KNearestNeighbors $e) : void {
        });

        $this->assertArrayNotHasKey('setup', $this->estimator->__serialize());

        $copy = unserialize(serialize($this->estimator));

        $this->assertInstanceOf(GridSearch::class, $copy);

        $this->assertArrayNotHasKey('setup', $copy->__serialize());
    }
}

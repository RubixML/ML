<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Base;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RunInSeparateProcess;
use Rubix\ML\DataType;
use Rubix\ML\GridSearch;
use Rubix\ML\EstimatorType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\CrossValidation\HoldOut;
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

        $ampBackend = new Amp(4);

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
}

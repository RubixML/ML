<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Base;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\DataType;
use Rubix\ML\BayesianSearch;
use Rubix\ML\EstimatorType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;

#[CoversClass(BayesianSearch::class)]
class BayesianSearchTest extends TestCase
{
    protected const int TRAIN_SIZE = 512;

    protected const int TEST_SIZE = 256;

    protected const float MIN_SCORE = 0.9;

    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected BayesianSearch $estimator;

    protected Accuracy $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'inner' => new Circle(x: 0.0, y: 0.0, scale: 1.0, noise: 0.5),
                'middle' => new Circle(x: 0.0, y: 0.0, scale: 5.0, noise: 1.0),
                'outer' => new Circle(x: 0.0, y: 0.0, scale: 10.0, noise: 2.0),
            ]
        );

        $this->estimator = new BayesianSearch(
            class: KNearestNeighbors::class,
            params: [
                [1, 3, 5, 10],
                [true, false],
                [
                    new Euclidean(),
                    new Manhattan(),
                ],
            ],
            metric: new FBeta(),
            validator: new HoldOut(0.2),
            maxTrials: 12
        );

        $this->metric = new Accuracy();

        srand(self::RANDOM_SEED);
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
                [1, 3, 5, 10],
                [true, false],
                [new Euclidean(), new Manhattan()],
            ],
            'initialTrials' => 10,
            'maxTrials' => 12,
            'quantile' => 0.25,
            'metric' => new FBeta(),
            'validator' => new HoldOut(0.2),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function trainPredictBest() : void
    {
        $this->estimator->setLogger(new BlackHole());

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
    public function scoresAreInTrialOrder() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $scores = $this->estimator->scores();

        $this->assertCount(12, $scores);
    }

    #[Test]
    public function resultsAreTableOfTrialsAndScores() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $scores = $this->estimator->scores();

        $this->assertNotEmpty($scores);

        $rows = $this->estimator->results()->toArray();

        $this->assertCount(12, $rows);

        $metric = new FBeta();

        foreach ($scores as $i => $score) {
            $row = $rows['Trial ' . ($i + 1)];

            $this->assertSame(
                ["{$metric}", 'params'],
                array_keys($row)
            );

            $this->assertSame(
                ['k', 'weighted', 'kernel'],
                array_keys($row['params'])
            );

            $this->assertSame((string) $score, $row["{$metric}"]);
        }
    }

    #[Test]
    public function resultsThrowBeforeTraining() : void
    {
        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('No trials have been run yet.');

        $this->estimator->results();
    }

    #[Test]
    public function fromNamedParams() : void
    {
        $estimator = BayesianSearch::fromNamedParams(
            KNearestNeighbors::class,
            params: [
                'kernel' => [new Manhattan(), new Euclidean()],
                'weighted' => [true],
                'k' => [1, 5, 10],
            ],
            metric: new FBeta(),
            validator: new HoldOut(0.2),
            maxTrials: 6
        );

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());

        $rows = $estimator->results()->toArray();

        $first = reset($rows);

        $this->assertSame(
            ['k', 'weighted', 'kernel'],
            array_keys($first['params'])
        );
    }

    #[Test]
    public function fromNamedParamsFillsDefaults() : void
    {
        $estimator = BayesianSearch::fromNamedParams(
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
            'initialTrials' => 10,
            'maxTrials' => 32,
            'quantile' => 0.25,
            'metric' => new FBeta(),
            'validator' => new KFold(5),
        ];

        $this->assertEquals($expected, $estimator->params());
    }

    #[Test]
    public function fillsEmptyTuplesWithDefaults() : void
    {
        $estimator = new BayesianSearch(
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
            'initialTrials' => 10,
            'maxTrials' => 32,
            'quantile' => 0.25,
            'metric' => new FBeta(),
            'validator' => new KFold(5),
        ];

        $this->assertEquals($expected, $estimator->params());
    }

    #[Test]
    public function fromNamedParamsRejectsUnknownParam() : void
    {
        $this->expectException(InvalidArgumentException::class);

        BayesianSearch::fromNamedParams(KNearestNeighbors::class, ['nope' => [true]]);
    }

    #[Test]
    public function rejectsScalarTuple() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new BayesianSearch(KNearestNeighbors::class, [10, [1, 5]]);
    }

    #[Test]
    public function rejectsUnorderedParams() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new BayesianSearch(KNearestNeighbors::class, ['k' => [1, 5]]);
    }

    #[Test]
    public function rejectsMaxTrialsLessThanOne() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new BayesianSearch(KNearestNeighbors::class, [[1]], maxTrials: 0);
    }

    #[Test]
    public function rejectsQuantileOutOfRange() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new BayesianSearch(KNearestNeighbors::class, [[1]], quantile: 1.0);
    }

    #[Test]
    public function rejectsNegativeInitialTrials() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new BayesianSearch(KNearestNeighbors::class, [[1]], initialTrials: -1);
    }

    #[Test]
    public function capsMaxTrialsToSpaceSize() : void
    {
        $estimator = new BayesianSearch(
            class: KNearestNeighbors::class,
            params: [
                [1, 3, 5, 10],
                [true],
                [new Euclidean()],
            ],
            maxTrials: 12
        );

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $estimator->train($training);

        $this->assertCount(4, $estimator->scores());
        $this->assertCount(4, $estimator->results()->toArray());
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

        // 12 max trials + 1 final best estimator
        $this->assertSame(13, $callCount);

        foreach ($types as $type) {
            $this->assertSame(KNearestNeighbors::class, $type);
        }
    }

    #[Test]
    #[TestDox('Setup closure is transient and excluded from serialization')]
    public function setupIsTransient() : void
    {
        $this->estimator->setup(function (KNearestNeighbors $e) : void {
        });

        $this->assertArrayNotHasKey('setup', $this->estimator->__serialize());

        $copy = unserialize(serialize($this->estimator));

        $this->assertInstanceOf(BayesianSearch::class, $copy);

        $this->assertArrayNotHasKey('setup', $copy->__serialize());
    }
}

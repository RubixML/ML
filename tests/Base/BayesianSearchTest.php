<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Base;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use Rubix\ML\DataType;
use Rubix\ML\BayesianSearch;
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
use Rubix\ML\Helpers\Params;
use PHPUnit\Framework\TestCase;

use function Rubix\ML\iterator_first;

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
            'metric' => new FBeta(),
            'validator' => new HoldOut(0.2),
            'maxTrials' => 12,
            'quantile' => 0.25,
            'startup' => 10,
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

        [$bestParams, $bestScore] = $this->estimator->best();

        $this->assertNotNull($bestParams);
        $this->assertNotNull($bestScore);

        foreach ($bestParams as $name => $value) {
            $this->assertSame($value, Params::toString($this->estimator->base()->params()[$name]));
        }
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

        $this->assertNotEmpty($this->estimator->scores());

        $progress = $this->estimator->results();

        $this->assertInstanceOf(Generator::class, $progress);

        $rows = iterator_to_array($progress);

        $this->assertCount(12, $rows);

        $metric = new FBeta();

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
    public function bestIsNullsBeforeTraining() : void
    {
        $this->assertSame([null, null], $this->estimator->best());
    }

    #[Test]
    public function bestMatchesTopPerformingTrial() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        [$bestParams, $bestScore] = $this->estimator->best();

        $metric = new FBeta();

        $first = iterator_first($this->estimator->results());

        $this->assertSame($first["{$metric}"], $bestScore);

        foreach ($bestParams as $name => $value) {
            $this->assertSame($value, $first[$name]);
        }
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

        $rows = iterator_to_array($estimator->results());

        $this->assertSame(
            ['k', 'weighted', 'kernel'],
            array_slice(array_keys($rows[0]), 0, 3)
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
            'metric' => new FBeta(),
            'validator' => new KFold(5),
            'maxTrials' => 32,
            'quantile' => 0.25,
            'startup' => 10,
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
            'metric' => new FBeta(),
            'validator' => new KFold(5),
            'maxTrials' => 32,
            'quantile' => 0.25,
            'startup' => 10,
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
    public function rejectsNegativeStartup() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new BayesianSearch(KNearestNeighbors::class, [[1]], startup: -1);
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
        $this->assertCount(4, iterator_to_array($estimator->results()));
    }
}

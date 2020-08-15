<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Wrapper;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\GridSearch;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Other\Loggers\BlackHole;
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;

/**
 * @group MetaEstimators
 * @covers \Rubix\ML\GridSearch
 */
class GridSearchTest extends TestCase
{
    protected const TRAIN_SIZE = 300;

    protected const TEST_SIZE = 10;

    protected const MIN_SCORE = 0.9;

    protected const RANDOM_SEED = 0;

    /**
     * @var \Rubix\ML\Datasets\Generators\Agglomerate
     */
    protected $generator;

    /**
     * @var \Rubix\ML\GridSearch
     */
    protected $estimator;

    /**
     * @var \Rubix\ML\CrossValidation\Metrics\Accuracy
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'inner' => new Circle(0.0, 0.0, 1.0, 0.05),
            'middle' => new Circle(0.0, 0.0, 5.0, 0.10),
            'outer' => new Circle(0.0, 0.0, 10.0, 0.15),
        ]);

        $this->estimator = new GridSearch(KNearestNeighbors::class, [
            [1, 3, 5], [true], [new Euclidean(), new Manhattan()],
        ], new FBeta(), new HoldOut(0.2));

        $this->metric = new Accuracy();

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(GridSearch::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Wrapper::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $this->assertEquals(DataType::all(), $this->estimator->compatibility());
    }

    /**
     * @test
     */
    public function params() : void
    {
        $expected = [
            'base' => KNearestNeighbors::class,
            'params' => [
                [1, 3, 5], [true], [new Euclidean(), new Manhattan()],
            ],
            'metric' => new FBeta(),
            'validator' => new HoldOut(0.2),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @test
     */
    public function trainPredictBest() : void
    {
        $this->estimator->setLogger(new BlackHole());
        $this->estimator->setBackend(new Serial());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $results = $this->estimator->results();

        $this->assertIsArray($results);
        $this->assertCount(6, $results);

        $best = $this->estimator->best();

        $this->assertIsArray($best);
        $this->assertCount(3, $best);
        $this->assertEquals($results[0][1], $best);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }
}

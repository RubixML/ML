<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\GridSearch;
use Rubix\ML\Persistable;
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

/**
 * @group MetaEstimators
 * @covers \Rubix\ML\GridSearch
 */
class GridSearchTest extends TestCase
{
    protected const TRAIN_SIZE = 512;

    protected const TEST_SIZE = 256;

    protected const MIN_SCORE = 0.9;

    protected const RANDOM_SEED = 0;

    /**
     * @var Agglomerate
     */
    protected $generator;

    /**
     * @var GridSearch
     */
    protected $estimator;

    /**
     * @var Accuracy
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'inner' => new Circle(0.0, 0.0, 1.0, 0.5),
            'middle' => new Circle(0.0, 0.0, 5.0, 1.0),
            'outer' => new Circle(0.0, 0.0, 10.0, 2.0),
        ]);

        $this->estimator = new GridSearch(KNearestNeighbors::class, [
            [1, 5, 10], [true], [new Euclidean(), new Manhattan()],
        ], new FBeta(), new HoldOut(0.2));

        $this->metric = new Accuracy();

        srand(self::RANDOM_SEED);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(GridSearch::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
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
     * @test
     */
    public function trainPredictBest() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);

        $expectedBest = [
            'k' => 10,
            'weighted' => true,
            'kernel' => new Manhattan(),
        ];

        $this->assertEquals($expectedBest, $this->estimator->base()->params());
    }
}

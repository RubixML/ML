<?php

namespace Rubix\ML\Tests\Clusterers;

use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;

/**
 * @group Clusterers
 * @covers \Rubix\ML\Clusterers\DBSCAN
 */
class DBSCANTest extends TestCase
{
    /**
     * The number of samples in the validation set.
     *
     * @var int
     */
    protected const TEST_SIZE = 512;

    /**
     * The minimum validation score required to pass the test.
     *
     * @var float
     */
    protected const MIN_SCORE = 0.9;

    /**
     * Constant used to see the random number generator.
     *
     * @var int
     */
    protected const RANDOM_SEED = 0;

    /**
     * @var Agglomerate
     */
    protected $generator;

    /**
     * @var DBSCAN
     */
    protected $estimator;

    /**
     * @var VMeasure
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'inner' => new Circle(0.0, 0.0, 1.0, 0.01),
            'middle' => new Circle(0.0, 0.0, 5.0, 0.05),
            'outer' => new Circle(0.0, 0.0, 10.0, 0.1),
        ]);

        $this->estimator = new DBSCAN(1.2, 20, new BallTree());

        $this->metric = new VMeasure();

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(DBSCAN::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function badRadius() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new DBSCAN(0.0);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(EstimatorType::clusterer(), $this->estimator->type());
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    /**
     * @test
     */
    public function params() : void
    {
        $expected = [
            'radius' => 1.2,
            'min density' => 20,
            'tree' => new BallTree(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @test
     */
    public function predict() : void
    {
        $testing = $this->generator->generate(self::TEST_SIZE);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    /**
     * @test
     */
    public function predictIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->predict(Unlabeled::quick([['bad']]));
    }
}

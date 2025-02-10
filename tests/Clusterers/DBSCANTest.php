<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;

#[Group('Clusterers')]
#[CoversClass(DBSCAN::class)]
class DBSCANTest extends TestCase
{
    /**
     * The number of samples in the validation set.
     */
    protected const int TEST_SIZE = 512;

    /**
     * The minimum validation score required to pass the test.
     */
    protected const float MIN_SCORE = 0.9;

    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected DBSCAN $estimator;

    protected VMeasure $metric;

    protected function setUp() : void
    {
        generators: $this->generator = new Agglomerate(
            [
                'inner' => new Circle(x: 0.0, y: 0.0, scale: 1.0, noise: 0.01),
                'middle' => new Circle(x: 0.0, y: 0.0, scale: 5.0, noise: 0.05),
                'outer' => new Circle(x: 0.0, y: 0.0, scale: 10.0, noise: 0.1),
            ]
        );

        $this->estimator = new DBSCAN(radius: 1.2, minDensity: 20, tree: new BallTree());

        $this->metric = new VMeasure();

        srand(self::RANDOM_SEED);
    }

    public function testBadRadius() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new DBSCAN(radius: 0.0);
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::clusterer(), $this->estimator->type());
    }

    public function testCompatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    public function testParams() : void
    {
        $expected = [
            'radius' => 1.2,
            'min density' => 20,
            'tree' => new BallTree(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    public function testPredict() : void
    {
        $testing = $this->generator->generate(self::TEST_SIZE);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function predictIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->predict(Unlabeled::quick(samples: [['bad']]));
    }
}

<?php

namespace Rubix\ML\Tests\Clusterers;

use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class DBSCANTest extends TestCase
{
    protected const TEST_SIZE = 300;
    protected const MIN_SCORE = 0.9;

    protected const RANDOM_SEED = 0;

    protected $generator;

    protected $estimator;

    protected $metric;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 32, 0], 30.),
            'green' => new Blob([0, 128, 0], 10.),
            'blue' => new Blob([0, 32, 255], 20.),
        ], [2, 3, 4]);

        $this->estimator = new DBSCAN(25.0, 50, new BallTree());

        $this->metric = new VMeasure();

        srand(self::RANDOM_SEED);
    }

    public function test_build_clusterer()
    {
        $this->assertInstanceOf(DBSCAN::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);

        $this->assertSame(Estimator::CLUSTERER, $this->estimator->type());

        $this->assertNotContains(DataType::CATEGORICAL, $this->estimator->compatibility());
        $this->assertContains(DataType::CONTINUOUS, $this->estimator->compatibility());
    }

    public function test_predict()
    {
        $testing = $this->generator->generate(self::TEST_SIZE);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function test_predict_incompatible()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->predict(Unlabeled::quick([['bad']]));
    }
}

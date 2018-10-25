<?php

namespace Rubix\ML\Tests\Clusterers;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

class DBSCANTest extends TestCase
{
    const TEST_SIZE = 300;
    const TOLERANCE = 3;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 3.),
            'green' => new Blob([0, 128, 0], 1.),
            'blue' => new Blob([0, 0, 255], 2.),
        ]);

        $this->estimator = new DBSCAN(10.0, 75, new Euclidean());
    }

    public function test_build_clusterer()
    {
        $this->assertInstanceOf(DBSCAN::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::CLUSTERER, $this->estimator->type());
    }

    public function test_train_predict()
    {
        $dataset = $this->generator->generate(self::TEST_SIZE);

        $predictions = $this->estimator->predict($dataset);

        $clusters = array_count_values($predictions);

        $this->assertEquals(self::TEST_SIZE / 3, $clusters[0], '', self::TOLERANCE);
        $this->assertEquals(self::TEST_SIZE / 3, $clusters[1], '', self::TOLERANCE);
        $this->assertEquals(self::TEST_SIZE / 3, $clusters[2], '', self::TOLERANCE);
    }
}

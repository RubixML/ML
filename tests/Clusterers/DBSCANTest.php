<?php

namespace Rubix\ML\Tests\Clusterers;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

class DBSCANTest extends TestCase
{
    const TOLERANCE = 3;

    protected $estimator;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = Labeled::load(dirname(__DIR__) . '/iris.dataset');

        $this->estimator = new DBSCAN(1.5, 5, new Euclidean());
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

    public function test_predict()
    {
        $results = $this->estimator->predict($this->dataset);

        $clusters = array_count_values($results);

        $this->assertEquals(50, $clusters[0], '', self::TOLERANCE);
        $this->assertEquals(50, $clusters[1], '', self::TOLERANCE);
    }
}

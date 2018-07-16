<?php

namespace Rubix\Tests\Clusterers;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\Clusterer;
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

class FuzzyCMeansTest extends TestCase
{
    protected $estimator;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = Labeled::restore(dirname(__DIR__) . '/iris.dataset');

        $this->estimator = new FuzzyCMeans(2, 2.0, new Euclidean(), 1e-4, 100);
    }

    public function test_build_clusterer()
    {
        $this->assertInstanceOf(FuzzyCMeans::class, $this->estimator);
        $this->assertInstanceOf(Clusterer::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->dataset);

        $results = $this->estimator->predict($this->dataset);

        $clusters = array_count_values($results);

        $this->assertEquals(50, $clusters[0], '', 3);
        $this->assertEquals(50, $clusters[1], '', 3);
    }
}

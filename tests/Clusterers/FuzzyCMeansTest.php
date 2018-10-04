<?php

namespace Rubix\ML\Tests\Clusterers;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class FuzzyCMeansTest extends TestCase
{
    const TOLERANCE = 3;
    
    protected $estimator;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = Labeled::load(dirname(__DIR__) . '/iris.dataset');

        $this->estimator = new FuzzyCMeans(2, 1.5, new Euclidean(), 1e-4, 100);
    }

    public function test_build_clusterer()
    {
        $this->assertInstanceOf(FuzzyCMeans::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::CLUSTERER, $this->estimator->type());
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->dataset);

        $results = $this->estimator->predict($this->dataset);

        $clusters = array_count_values($results);

        $this->assertEquals(50, $clusters[0], '', self::TOLERANCE);
        $this->assertEquals(50, $clusters[1], '', self::TOLERANCE);
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict($this->dataset);
    }
}

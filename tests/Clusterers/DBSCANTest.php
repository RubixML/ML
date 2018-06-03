<?php

use Rubix\Engine\Estimator;
use Rubix\Engine\Unsupervised;
use Rubix\Engine\Clusterers\DBSCAN;
use Rubix\Engine\Datasets\Unlabeled;
use Rubix\Engine\Clusterers\Clusterer;
use Rubix\Engine\Metrics\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

class DBSCANTest extends TestCase
{
    protected $estimator;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
            [2.771244718, 1.784783929], [1.728571309, 1.169761413],
            [3.678319846, 2.812813570], [3.961043357, 2.619950320],
            [2.999208922, 2.209014212], [2.345634564, 1.345634563],
            [1.678967899, 1.345634566], [3.234523455, 2.123411234],
            [3.456745685, 2.678960008], [2.234523463, 2.345633224],
            [7.497545867, 3.162953546], [9.002203261, 3.339047188],
            [7.444542326, 0.476683375], [10.12493903, 3.234550982],
            [6.642287351, 3.319983761], [7.670678677, 3.234556477],
            [9.345234522, 3.768960060], [7.234523457, 0.736747567],
            [10.56785567, 3.123412342], [6.456749570, 3.324523456],
        ]);

        $this->estimator = new DBSCAN(2.5, 5, new Euclidean());
    }

    public function test_build_DBSCAN_clusterer()
    {
        $this->assertInstanceOf(DBSCAN::class, $this->estimator);
        $this->assertInstanceOf(Clusterer::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Unsupervised::class, $this->estimator);
    }

    public function test_cluster()
    {
        $this->estimator->train($this->dataset);

        $results = $this->estimator->predict($this->dataset);

        $clusters = array_count_values($results);

        $this->assertEquals(10, $clusters[0]);
        $this->assertEquals(9, $clusters[1]);
        $this->assertEquals(1, $clusters['noise']);
    }
}

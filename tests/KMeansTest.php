<?php

use Rubix\Engine\KMeans;
use Rubix\Engine\Datasets\Unsupervised;
use Rubix\Engine\Graph\DistanceFunctions\Euclidean;
use PHPUnit\Framework\TestCase;

class KMeansTest extends TestCase
{
    protected $estimator;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = Unsupervised::fromIterator([
            [2.771244718, 1.784783929],
            [1.728571309, 1.169761413],
            [3.678319846, 2.812813570],
            [3.961043357, 2.619950320],
            [2.999208922, 2.209014212],
            [2.345634564, 1.345634563],
            [1.678967899, 1.345634566],
            [3.234523455, 2.123411234],
            [3.456745685, 2.678960008],
            [2.234523463, 2.345633224],
            [7.497545867, 3.162953546],
            [9.002203261, 3.339047188],
            [7.444542326, 0.476683375],
            [10.12493903, 3.234550982],
            [6.642287351, 3.319983761],
            [7.670678677, 3.234556477],
            [9.345234522, 3.768960060],
            [7.234523457, 0.736747567],
            [10.56785567, 3.123412342],
            [6.456749570, 3.324523456],
        ]);

        $this->estimator = new KMeans(2, new Euclidean(), 500);
    }

    public function test_build_k_means_clusterer()
    {
        $this->assertInstanceOf(KMeans::class, $this->estimator);
    }

    public function test_cluster_samples()
    {
        $clusters = $this->estimator->cluster($this->dataset);

        $this->assertEquals(10, count($clusters[0]));
        $this->assertEquals(10, count($clusters[1]));
    }
}

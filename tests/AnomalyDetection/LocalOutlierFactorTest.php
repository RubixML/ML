<?php

use Rubix\ML\Online;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\AnomalyDetectors\Detector;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\AnomalyDetectors\LocalOutlierFactor;
use PHPUnit\Framework\TestCase;

class LocalOutlierFactorTest extends TestCase
{
    protected $estimator;

    protected $dataset;

    public function setUp()
    {
        $this->clean = new Unlabeled([
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

        $this->dirty = new Unlabeled([
            [10.032273011, 2.469057128], [3.612394031, 2.645327321],
            [1.0177273113, 4.727491941], [9.293847293, 3.293847293],
        ]);

        $this->estimator = new LocalOutlierFactor(5, 4, 0.5, new Euclidean());
    }

    public function test_build_local_outlier_factor_detector()
    {
        $this->assertInstanceOf(LocalOutlierFactor::class, $this->estimator);
        $this->assertInstanceOf(Detector::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_predict()
    {
        $this->estimator->train($this->clean);

        $results = $this->estimator->predict($this->dirty);

        $this->assertEquals([1, 0, 1, 0], $results);
    }

    public function test_predict_proba()
    {
        $this->estimator->train($this->clean);

        $results = $this->estimator->proba($this->dirty);

        $this->assertGreaterThan(0.5, $results[0]);
        $this->assertLessThanOrEqual(0.5, $results[1]);
        $this->assertGreaterThan(0.5, $results[2]);
        $this->assertLessThanOrEqual(0.5, $results[3]);
    }
}

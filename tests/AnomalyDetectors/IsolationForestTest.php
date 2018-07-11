<?php

namespace Rubix\Tests\AnomalyDetectors;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\AnomalyDetectors\Detector;
use Rubix\ML\AnomalyDetectors\IsolationForest;
use PHPUnit\Framework\TestCase;

class IsolationForestTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = new Unlabeled([
            [1.582984092, 1.290384209], [2.928349282, 2.020312931],
            [1.805983042, 1.348957322], [3.189832223, 2.928374922],
            [1.934283999, 2.109303483], [1.883923988, 1.487293048],
            [2.452736473, 1.938837472], [1.387482914, 1.463728342],
            [2.771244718, 1.784783929], [1.728571309, 1.169761413],
            [3.678319846, 2.812813576], [3.961043357, 2.619950321],
            [2.999208922, 2.209014212], [2.345634564, 1.345634563],
            [1.678967899, 1.345634566], [3.234523455, 2.123411234],
            [3.456745685, 2.678960008], [2.234523463, 2.345633224],
            [7.497545867, 3.162953546], [9.002203261, 3.339047188],
            [7.444542326, 2.776683375], [10.12493903, 3.234550982],
            [6.642287351, 3.319983761], [7.670678677, 3.234556477],
            [9.345234522, 3.768960061], [7.234523457, 1.936747567],
            [10.56785567, 3.123412342], [6.456749570, 3.324523456],
            [7.239482342, 3.328743984], [9.295829348, 3.023488922],
            [6.993423428, 3.149842333], [7.489023983, 3.690345001],
            [10.34798234, 3.290349822], [8.128312313, 2.849098342],
            [7.283049823, 3.239480239], [9.203948022, 2.023409288],
        ]);

        $this->testing = new Labeled([
            [10.932273011, 2.269057128], [3.612394031, 2.645327321],
            [1.0177273113, 4.727491941], [9.293847293, 3.293847293],
        ], [
            1, 0, 1, 0,
        ]);

        $this->estimator = new IsolationForest(500, 0.8, 0.5);
    }

    public function test_build_detector()
    {
        $this->assertInstanceOf(IsolationForest::class, $this->estimator);
        $this->assertInstanceOf(Detector::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        $results = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $results[0]);
        $this->assertEquals($this->testing->label(1), $results[1]);
        $this->assertEquals($this->testing->label(2), $results[2]);
        $this->assertEquals($this->testing->label(3), $results[3]);
    }

    public function test_predict_proba()
    {
        $this->estimator->train($this->training);

        $results = $this->estimator->proba($this->testing);

        $this->assertGreaterThan(0.50, $results[0]);
        $this->assertLessThanOrEqual(0.50, $results[1]);
        $this->assertGreaterThan(0.50, $results[2]);
        $this->assertLessThanOrEqual(0.50, $results[3]);
    }
}

<?php

use Rubix\Engine\DecisionForest;
use Rubix\Engine\Datasets\Supervised;
use PHPUnit\Framework\TestCase;

class DecisionForestTest extends TestCase
{
    protected $estimator;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = Supervised::fromIterator([
            [2.771244718, 1.784783929, 'female'],
            [1.728571309, 1.169761413, 'female'],
            [3.678319846, 2.812813570, 'female'],
            [3.961043357, 2.619950320, 'female'],
            [2.999208922, 2.209014212, 'female'],
            [2.345634564, 1.345634563, 'female'],
            [1.678967899, 1.345634566, 'female'],
            [3.234523455, 2.123411234, 'female'],
            [3.456745685, 2.678960008, 'female'],
            [2.234523463, 2.345633224, 'female'],
            [7.497545867, 3.162953546, 'male'],
            [9.002203261, 3.339047188, 'male'],
            [7.444542326, 0.476683375, 'male'],
            [10.12493903, 3.234550982, 'male'],
            [6.642287351, 3.319983761, 'male'],
            [7.670678677, 3.234556477, 'male'],
            [9.345234522, 3.768960060, 'male'],
            [7.234523457, 0.736747567, 'male'],
            [10.56785567, 3.123412342, 'male'],
            [6.456749570, 3.324523456, 'male'],
        ]);

        $this->estimator = new DecisionForest(10, 0.3, 3, 30);

        $this->estimator->train($this->dataset);
    }

    public function test_create_tree()
    {
        $this->assertTrue($this->estimator instanceof DecisionForest);
    }

    public function test_make_prediction()
    {
        $prediction = $this->estimator->predict([7.5, 3.2]);

        $this->assertEquals('male', $prediction->outcome());
    }
}

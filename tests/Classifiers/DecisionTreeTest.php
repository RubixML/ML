<?php

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\Multiclass;
use Rubix\ML\Classifiers\Classifier;
use Rubix\ML\Classifiers\DecisionTree;
use Rubix\ML\Probabilistic;

use PHPUnit\Framework\TestCase;

class DecisionTreeTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $samples = [
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
        ];

        $labels = [
            'female','female', 'female', 'female', 'female','female', 'female',
            'female', 'female', 'female', 'male', 'male', 'male', 'male',
            'male', 'male', 'male', 'male', 'male', 'male',
        ];

        $this->training = new Labeled($samples, $labels);

        $this->testing = new Labeled([[7.1929367, 3.52848298]], ['male']);

        $this->estimator = new DecisionTree(10, 5, 1e-4);
    }

    public function test_create_tree()
    {
        $this->assertInstanceOf(DecisionTree::class, $this->estimator);
        $this->assertInstanceOf(Classifier::class, $this->estimator);
        $this->assertInstanceOf(Multiclass::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals('male', $predictions[0]);
    }

    public function test_predict_proba()
    {
        $this->estimator->train($this->training);

        $probabilities = $this->estimator->proba($this->testing);

        $this->assertGreaterThan(0.5, $probabilities[0]['male']);
        $this->assertLessThan(0.5, $probabilities[0]['female']);
    }
}

<?php

namespace Rubix\Tests\Classifiers;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\Multiclass;
use Rubix\ML\Classifiers\Classifier;
use Rubix\ML\Classifiers\NaiveBayes;
use PHPUnit\Framework\TestCase;

class NaiveBayesTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $samples = [
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
        ];

        $labels = [
            'female','female', 'female', 'female', 'female','female', 'female',
            'female', 'female', 'female', 'female', 'female', 'female', 'female',
            'female', 'female', 'female', 'female', 'male', 'male', 'male',
            'male', 'male', 'male', 'male', 'male', 'male', 'male', 'male',
            'male', 'male', 'male', 'male', 'male', 'male', 'male',
        ];

        $this->training = new Labeled($samples, $labels);

        $this->testing = new Labeled([
            [7.1929367, 3.52848298], [2.23429374, 1.71279293],
        ], [
            'male', 'female',
        ]);

        $this->estimator = new NaiveBayes();
    }

    public function test_create_classifier()
    {
        $this->assertInstanceOf(NaiveBayes::class, $this->estimator);
        $this->assertInstanceOf(Classifier::class, $this->estimator);
        $this->assertInstanceOf(Multiclass::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->training->randomize();

        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals('male', $predictions[0]);
        $this->assertEquals('female', $predictions[1]);
    }
}

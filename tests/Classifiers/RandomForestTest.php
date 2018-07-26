<?php

namespace Rubix\Tests\Classifiers;

use Rubix\ML\Ensemble;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\ExtraTreeClassifier;
use Rubix\ML\Classifiers\Multiclass;
use Rubix\ML\Classifiers\Classifier;
use Rubix\ML\Classifiers\RandomForest;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class RandomForestTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = Labeled::restore(dirname(__DIR__) . '/iris.dataset');

        $this->testing = $this->training->randomize()->head(3);

        $this->estimator = new RandomForest(100, 0.3, 10, 5, 4, 1e-3, ExtraTreeClassifier::class);
    }

    public function test_build_classifier()
    {
        $this->assertInstanceOf(RandomForest::class, $this->estimator);
        $this->assertInstanceOf(Classifier::class, $this->estimator);
        $this->assertInstanceOf(Multiclass::class, $this->estimator);
        $this->assertInstanceOf(Ensemble::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0]);
        $this->assertEquals($this->testing->label(1), $predictions[1]);
        $this->assertEquals($this->testing->label(2), $predictions[2]);

        $probabilities = $this->estimator->proba($this->testing);

        $this->assertGreaterThanOrEqual(0.5, $probabilities[0][$this->testing->label(0)]);
        $this->assertGreaterThanOrEqual(0.5, $probabilities[1][$this->testing->label(1)]);
        $this->assertGreaterThanOrEqual(0.5, $probabilities[2][$this->testing->label(2)]);
    }

    public function test_train_with_unlabeled()
    {
        $dataset = new Unlabeled([['bad']]);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train($dataset);
    }
}

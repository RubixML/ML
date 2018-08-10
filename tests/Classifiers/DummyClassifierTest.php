<?php

namespace Rubix\Tests\Classifiers;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Other\Strategies\PopularityContest;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class DummyClassifierTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = Labeled::restore(dirname(__DIR__) . '/iris.dataset');

        $this->testing = $this->training->randomize()->head(3);

        $this->estimator = new DummyClassifier(new PopularityContest());
    }

    public function test_build_classifier()
    {
        $this->assertInstanceOf(DummyClassifier::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::CLASSIFIER, $this->estimator->type());
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertContains($predictions[0], $this->training->possibleOutcomes());
        $this->assertContains($predictions[1], $this->training->possibleOutcomes());
        $this->assertContains($predictions[2], $this->training->possibleOutcomes());
    }

    public function test_train_with_unlabeled()
    {
        $dataset = new Unlabeled([['bad']]);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train($dataset);
    }
}

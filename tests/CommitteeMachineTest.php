<?php

namespace Rubix\Tests;

use Rubix\ML\Ensemble;
use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\CommitteeMachine;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Classifiers\ClassificationTree;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class CommitteeMachineTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = Labeled::restore(__DIR__ . '/iris.dataset');

        $this->testing = $this->training->randomize()->head(3);

        $this->estimator = new CommitteeMachine([
            new ClassificationTree(10, 3, 4),
            new KDNeighbors(5, 20, new Euclidean()),
            new GaussianNB(),
        ]);
    }

    public function test_build_meta_estimator()
    {
        $this->assertInstanceOf(CommitteeMachine::class, $this->estimator);
        $this->assertInstanceOf(MetaEstimator::class, $this->estimator);
        $this->assertInstanceOf(Ensemble::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0]);
        $this->assertEquals($this->testing->label(1), $predictions[1]);
        $this->assertEquals($this->testing->label(2), $predictions[2]);
    }
}

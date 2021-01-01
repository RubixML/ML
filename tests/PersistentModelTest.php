<?php

namespace Rubix\ML\Tests;

use Rubix\ML\AnomalyDetectors\Scoring;
use Rubix\ML\Learner;
use Rubix\ML\Wrapper;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Classifiers\DummyClassifier;
use PHPUnit\Framework\TestCase;

/**
 * @group MetaEstimators
 * @covers \Rubix\ML\PersistentModel
 */
class PersistentModelTest extends TestCase
{
    /**
     * @var \Rubix\ML\PersistentModel
     */
    protected $estimator;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->estimator = new PersistentModel(new DummyClassifier(), new Filesystem('test.model'));
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(PersistentModel::class, $this->estimator);
        $this->assertInstanceOf(Wrapper::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Scoring::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $this->assertEquals(DataType::all(), $this->estimator->compatibility());
    }

    /**
     * @test
     */
    public function params() : void
    {
        $expected = [
            'base' => new DummyClassifier(),
            'persister' => new Filesystem('test.model'),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }
}

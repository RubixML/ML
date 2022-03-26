<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\PersistentModel;
use Rubix\ML\Serializers\RBX;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\AnomalyDetectors\Scoring;
use Rubix\ML\Classifiers\GaussianNB;
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
        $this->estimator = new PersistentModel(new GaussianNB(), new Filesystem('test.model'), new RBX());
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(PersistentModel::class, $this->estimator);
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
        $this->assertEquals([DataType::continuous()], $this->estimator->compatibility());
    }

    /**
     * @test
     */
    public function params() : void
    {
        $expected = [
            'base' => new GaussianNB(),
            'persister' => new Filesystem('test.model'),
            'serializer' => new RBX(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }
}

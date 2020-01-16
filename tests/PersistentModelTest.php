<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Wrapper;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Probabilistic;
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
     * @var \Rubix\ML\Persisters\Persister
     */
    protected $persister;

    /**
     * @var \Rubix\ML\PersistentModel
     */
    protected $estimator;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->persister = $this->createMock(Filesystem::class);

        $this->estimator = new PersistentModel(new DummyClassifier(), $this->persister);
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(PersistentModel::class, $this->estimator);
        $this->assertInstanceOf(Wrapper::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertSame(Estimator::CLASSIFIER, $this->estimator->type());
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $this->assertEquals(DataType::all(), $this->estimator->compatibility());
    }
}

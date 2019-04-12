<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Wrapper;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Probabilistic;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Loggers\BlackHole;
use Rubix\ML\Classifiers\DummyClassifier;
use PHPUnit\Framework\TestCase;

class PersistentModelTest extends TestCase
{
    protected $persister;

    protected $estimator;

    public function setUp()
    {
        $this->persister = $this->createMock(Filesystem::class);

        $this->estimator = new PersistentModel(new DummyClassifier(), $this->persister);

        $this->estimator->setLogger(new BlackHole());
    }

    public function test_build_meta_estimator()
    {
        $this->assertInstanceOf(PersistentModel::class, $this->estimator);
        $this->assertInstanceOf(Wrapper::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);

        $this->assertSame(Estimator::CLASSIFIER, $this->estimator->type());

        $this->assertContains(DataType::CATEGORICAL, $this->estimator->compatibility());
        $this->assertContains(DataType::CONTINUOUS, $this->estimator->compatibility());
    }
}

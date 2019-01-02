<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Wrapper;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Classifiers\DummyClassifier;
use PHPUnit\Framework\TestCase;

class PersistentModelTest extends TestCase
{
    protected $persister;

    protected $model;

    public function setUp()
    {
        $this->persister = $this->createMock(Filesystem::class);

        $this->model = new PersistentModel(new DummyClassifier(), $this->persister);
    }

    public function test_build_meta_estimator()
    {
        $this->assertInstanceOf(PersistentModel::class, $this->model);
        $this->assertInstanceOf(Wrapper::class, $this->model);
        $this->assertInstanceOf(Probabilistic::class, $this->model);
        $this->assertInstanceOf(Verbose::class, $this->model);
        $this->assertInstanceOf(Estimator::class, $this->model);
    }
}

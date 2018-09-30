<?php

namespace Rubix\Tests;

use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\PersistentModel;
use Rubix\ML\Classifiers\DummyClassifier;
use PHPUnit\Framework\TestCase;

class PersistentModelTest extends TestCase
{
    protected $model;

    public function setUp()
    {
        $this->model = new PersistentModel(new DummyClassifier());
    }

    public function test_build_meta_estimator()
    {
        $this->assertInstanceOf(PersistentModel::class, $this->model);
        $this->assertInstanceOf(MetaEstimator::class, $this->model);
    }

    public function test_save_and_restore()
    {
        $this->assertFalse(file_exists(__DIR__ . '/test.model'));

        $this->model->save(__DIR__ . '/test.model', true);

        $this->assertFileExists(__DIR__ . '/test.model');

        $model = PersistentModel::restore(__DIR__ . '/test.model');

        $this->assertInstanceOf(PersistentModel::class, $model);
        $this->assertInstanceOf(Persistable::class, $model->estimator());

        $this->assertTrue(unlink(__DIR__ . '/test.model'));
    }
}

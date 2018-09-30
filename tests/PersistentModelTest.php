<?php

namespace Rubix\Tests;

use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\PersistentModel;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Persisters\Filesystem;
use PHPUnit\Framework\TestCase;

class PersistentModelTest extends TestCase
{
    protected $model;

    public function setUp()
    {
        $this->model = new PersistentModel(new DummyClassifier(), new Filesystem(__DIR__ . '/test.model'));
    }

    public function test_build_meta_estimator()
    {
        $this->assertInstanceOf(PersistentModel::class, $this->model);
        $this->assertInstanceOf(MetaEstimator::class, $this->model);
    }

    public function test_save_restore_and_delete()
    {
        $this->assertFalse(file_exists(__DIR__ . '/test.model'));

        $this->model->save();

        $this->assertFileExists(__DIR__ . '/test.model');

        $model = PersistentModel::restore(new Filesystem(__DIR__ . '/test.model'));

        $this->assertInstanceOf(PersistentModel::class, $model);
        $this->assertInstanceOf(Persistable::class, $model->estimator());

        $this->model->delete();

        $this->assertFalse(file_exists(__DIR__ . '/test.model'));
    }
}

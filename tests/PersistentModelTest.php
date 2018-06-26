<?php

use Rubix\ML\Persistable;
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

    public function test_create_tree()
    {
        $this->assertInstanceOf(PersistentModel::class, $this->model);
    }

    public function test_save_model()
    {
        $this->assertFalse(file_exists(__DIR__ . '/test.model'));

        $this->model->save(__DIR__ . '/test.model');

        $this->assertFileExists(__DIR__ . '/test.model');
    }

    public function test_restore_model()
    {
        $model = PersistentModel::restore(__DIR__ . '/test.model');

        $this->assertInstanceOf(PersistentModel::class, $model);
        $this->assertInstanceOf(Persistable::class, $model->estimator());

        $this->assertTrue(unlink(__DIR__ . '/test.model'));
    }
}

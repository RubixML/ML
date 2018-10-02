<?php

namespace Rubix\Tests\Persisters;

use Rubix\ML\Persistable;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Classifiers\DummyClassifier;
use PHPUnit\Framework\TestCase;

class FilesystemTest extends TestCase
{
    protected $persistable;

    protected $persister;

    public function setUp()
    {
        $this->persistable = new DummyClassifier();

        $this->persister = new Filesystem(__DIR__ . '/test.model', true);
    }

    public function test_build_persister()
    {
        $this->assertInstanceOf(Filesystem::class, $this->persister);
        $this->assertInstanceOf(Persister::class, $this->persister);
    }

    public function test_save_load_and_delete()
    {
        $this->assertFalse(file_exists(__DIR__ . '/test.model'));

        $this->persister->save($this->persistable);

        $this->assertFileExists(__DIR__ . '/test.model');

        $model = $this->persister->load();

        $this->assertInstanceOf(Persistable::class, $model);

        $this->persister->delete();

        $this->assertFalse(file_exists(__DIR__ . '/test.model'));
    }
}

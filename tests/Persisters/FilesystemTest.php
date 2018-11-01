<?php

namespace Rubix\ML\Tests\Persisters;

use Rubix\ML\Persistable;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Classifiers\DummyClassifier;
use PHPUnit\Framework\TestCase;

class FilesystemTest extends TestCase
{
    protected $persistable;

    protected $persister;

    protected $path;

    public function setUp()
    {
        $this->path = __DIR__ . '/test.model';

        $this->persistable = new DummyClassifier();

        $this->persister = new Filesystem($this->path, 0);
    }

    public function test_build_persister()
    {
        $this->assertInstanceOf(Filesystem::class, $this->persister);
        $this->assertInstanceOf(Persister::class, $this->persister);
    }

    public function test_save_load_and_delete()
    {
        $this->assertFalse(file_exists($this->path));

        $this->persister->save($this->persistable);

        $this->assertFileExists($this->path);

        $model = $this->persister->load();

        $this->assertInstanceOf(DummyClassifier::class, $model);
        $this->assertInstanceOf(Persistable::class, $model);

        unlink($this->path);

        $this->assertFalse(file_exists($this->path));
    }
}

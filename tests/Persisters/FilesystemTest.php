<?php

namespace Rubix\ML\Tests\Persisters;

use Rubix\ML\Persistable;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Persisters\Serializers\Native;
use PHPUnit\Framework\TestCase;

class FilesystemTest extends TestCase
{
    /**
     * @var \Rubix\ML\Persistable
     */
    protected $persistable;

    /**
     * @var \Rubix\ML\Persisters\Filesystem
     */
    protected $persister;

    /**
     * @var string
     */
    protected $path;

    public function setUp() : void
    {
        $this->path = __DIR__ . '/test.model';

        $this->persistable = new DummyClassifier();

        $this->persister = new Filesystem($this->path, true, new Native());
    }

    public function test_build_persister() : void
    {
        $this->assertInstanceOf(Filesystem::class, $this->persister);
        $this->assertInstanceOf(Persister::class, $this->persister);
    }

    public function test_save_and_load() : void
    {
        $this->assertFalse(file_exists($this->path));

        $this->persister->save($this->persistable);
        $this->persister->save($this->persistable);

        $this->assertFileExists($this->path);

        $model = $this->persister->load();

        $this->assertInstanceOf(DummyClassifier::class, $model);
        $this->assertInstanceOf(Persistable::class, $model);

        foreach (glob("$this->path.*.old") ?: [] as $filename) {
            unlink($filename);
        }

        unlink($this->path);

        $this->assertFalse(file_exists($this->path));
    }
}

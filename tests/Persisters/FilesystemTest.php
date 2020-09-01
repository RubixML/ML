<?php

namespace Rubix\ML\Tests\Persisters;

use Rubix\ML\Other\Helpers\Storage;
use Rubix\ML\Persistable;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Persisters\Serializers\Native;
use PHPUnit\Framework\TestCase;

/**
 * @group Persisters
 * @covers \Rubix\ML\Persisters\Filesystem
 */
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

    /**
     * @var \League\Flysystem\FilesystemInterface
     */
    protected $filesystem;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->path = __DIR__ . '/test.model';
        $this->persistable = new DummyClassifier();
        $this->filesystem = Storage::memory();
        $this->persister = new Filesystem($this->path, true, new Native(), $this->filesystem);
    }

    /**
     * @after
     */
    protected function tearDown() : void
    {
        if ($this->filesystem->has($this->path)) {
            $this->filesystem->delete($this->path);
        }
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Filesystem::class, $this->persister);
        $this->assertInstanceOf(Persister::class, $this->persister);
    }

    /**
     * @test
     */
    public function saveLoad() : void
    {
        $this->persister->save($this->persistable);
        $this->assertTrue(
            $this->filesystem->has($this->path),
            'Persistable was not saved as expected'
        );

        $model = $this->persister->load();

        $this->assertInstanceOf(DummyClassifier::class, $model);
        $this->assertInstanceOf(Persistable::class, $model);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse(
            $this->filesystem->has($this->path),
            sprintf('File: %s already exists', $this->path)
        );
    }
}

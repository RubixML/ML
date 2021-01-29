<?php

namespace Rubix\ML\Tests\Persisters;

use Rubix\ML\Persistable;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Persisters\Serializers\RBX;
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
     * @before
     */
    protected function setUp() : void
    {
        $this->path = __DIR__ . '/test.model';

        $this->persistable = new DummyClassifier();

        $this->persister = new Filesystem($this->path, true, new RBX());
    }

    /**
     * @after
     */
    protected function tearDown() : void
    {
        if (file_exists($this->path)) {
            unlink($this->path);
        }

        foreach (glob("$this->path.*.old") ?: [] as $filename) {
            unlink($filename);
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

        $this->assertFileExists($this->path);

        $model = $this->persister->load();

        $this->assertInstanceOf(DummyClassifier::class, $model);
        $this->assertInstanceOf(Persistable::class, $model);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFileNotExists($this->path);
    }
}

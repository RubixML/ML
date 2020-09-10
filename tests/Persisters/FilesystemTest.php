<?php

namespace Rubix\ML\Tests\Persisters;

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
     * The path to the test file.
     *
     * @var string
     */
    const PATH = __DIR__ . '/test.persistable';

    /**
     * @var \Rubix\ML\Persistable
     */
    protected $persistable;

    /**
     * @var \Rubix\ML\Persisters\Filesystem
     */
    protected $persister;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->persistable = new DummyClassifier();

        $this->persister = new Filesystem(self::PATH, true, new Native());
    }

    /**
     * @after
     */
    protected function tearDown() : void
    {
        if (file_exists(self::PATH)) {
            unlink(self::PATH);
        }

        foreach (glob(self::PATH . '.*.old') ?: [] as $filename) {
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

        $this->assertFileExists(self::PATH);

        $persistable = $this->persister->load();

        $this->assertInstanceOf(DummyClassifier::class, $persistable);
        $this->assertInstanceOf(Persistable::class, $persistable);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFileNotExists(self::PATH);
    }
}

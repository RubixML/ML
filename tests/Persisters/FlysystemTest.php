<?php

namespace Rubix\ML\Tests\Persisters;

use Rubix\ML\Persistable;
use Rubix\ML\Persisters\Flysystem;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Persisters\Serializers\RBX;
use Rubix\ML\Exceptions\RuntimeException;
use League\Flysystem\Filesystem;
use League\Flysystem\FilesystemOperator;
use League\Flysystem\InMemory\InMemoryFilesystemAdapter;
use League\Flysystem\UnableToWriteFile;
use League\Flysystem\UnableToMoveFile;
use PHPUnit\Framework\TestCase;

/**
 * @group Persisters
 * @covers \Rubix\ML\Persisters\Flysystem
 */
class FlysystemTest extends TestCase
{
    /**
     * The path to the test file.
     *
     * @var string
     */
    const PATH = __DIR__ . '/path/to/test.persistable';

    /**
     * @var \League\Flysystem\FilesystemOperator
     */
    protected $filesystem;

    /**
     * @var \Rubix\ML\Persistable
     */
    protected $persistable;

    /**
     * @var \Rubix\ML\Persisters\Flysystem
     */
    protected $persister;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->filesystem = new Filesystem(new InMemoryFilesystemAdapter());

        $this->persistable = new DummyClassifier();

        $this->persister = new Flysystem(self::PATH, $this->filesystem, false, new RBX());
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Flysystem::class, $this->persister);
        $this->assertInstanceOf(Persister::class, $this->persister);
    }

    /**
     * @test
     */
    public function saveLoad() : void
    {
        $this->persister->save($this->persistable);

        $this->assertTrue($this->filesystem->fileExists(self::PATH));

        $persistable = $this->persister->load();

        $this->assertInstanceOf(DummyClassifier::class, $persistable);
        $this->assertInstanceOf(Persistable::class, $persistable);
    }

    /**
     * @test
     */
    public function saveMethodWhenFilesystemWriteFails() : void
    {
        $filesystem = $this->createMock(FilesystemOperator::class);

        $filesystem->method('write')
            ->with(self::PATH)
            ->willThrowException(new UnableToWriteFile());

        $this->persister = new Flysystem(self::PATH, $filesystem);

        $this->expectException(RuntimeException::class);

        $this->persister->save($this->persistable);
    }

    /**
     * @test
     */
    public function saveMethodWithHistoryDisabled() : void
    {
        $directory = dirname(self::PATH);

        $this->persister = new Flysystem(self::PATH, $this->filesystem, false);

        $this->persister->save($this->persistable);

        $this->assertCount(1, $this->filesystem->listContents($directory));
        $this->assertTrue($this->filesystem->fileExists(self::PATH));

        $this->persister->save($this->persistable);

        $this->assertCount(1, $this->filesystem->listContents($directory));
        $this->assertTrue($this->filesystem->fileExists(self::PATH));
    }

    /**
     * @test
     */
    public function saveMethodWithHistoryEnabled() : void
    {
        $directory = dirname(self::PATH);

        $this->persister = new Flysystem(self::PATH, $this->filesystem, true);

        $this->persister->save($this->persistable);

        $this->assertTrue($this->filesystem->fileExists(self::PATH));

        $this->persister->save($this->persistable);

        $files = $this->filesystem->listContents($directory);

        $this->assertCount(2, $files);
    }

    /**
     * @test
     */
    public function saveMethodWhenHistoryCreationFails() : void
    {
        $mock = $this->createMock(FilesystemOperator::class);

        $mock->expects($this->any())
            ->method('fileExists')
            ->will($this->onConsecutiveCalls(true, true, false));

        $mock->expects($this->any())
            ->method('move')
            ->willThrowException(new UnableToMoveFile());

        $this->persister = new Flysystem(self::PATH, $mock, true);

        $this->expectException(RuntimeException::class);

        $this->persister->save($this->persistable);
    }

    /**
     * @test
     */
    public function loadMethodWhenTargetNotExists() : void
    {
        $this->expectException(RuntimeException::class);

        $this->persister->load();
    }

    /**
     * @test
     */
    public function loadMethodWhenTargetIsEmpty() : void
    {
        $this->filesystem->write(self::PATH, '');

        $this->expectException(RuntimeException::class);

        $this->persister->load();
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->filesystem->fileExists(self::PATH));
    }
}

<?php

namespace Rubix\ML\Tests\Other\Helpers;

use League\Flysystem\Adapter\Local;
use League\Flysystem\Filesystem;
use League\Flysystem\FilesystemInterface;
use League\Flysystem\Memory\MemoryAdapter;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Other\Helpers\Storage;

/**
 * @group Helpers
 * @covers \Rubix\ML\Other\Helpers\Storage
 */
class StorageTest extends TestCase
{
    /**
     * @test
     */
    public function local() : void
    {
        /** @var Filesystem */
        $filesystem = Storage::local();

        $this->assertInstanceOf(FilesystemInterface::class, $filesystem);
        $this->assertInstanceOf(Filesystem::class, $filesystem);
        $this->assertInstanceOf(Local::class, $filesystem->getAdapter());
    }

    /**
     * @test
     */
    public function memory() : void
    {
        /** @var Filesystem */
        $filesystem = Storage::memory();

        $this->assertInstanceOf(FilesystemInterface::class, $filesystem);
        $this->assertInstanceOf(Filesystem::class, $filesystem);
        $this->assertInstanceOf(MemoryAdapter::class, $filesystem->getAdapter());
    }
}

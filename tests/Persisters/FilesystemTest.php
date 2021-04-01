<?php

namespace Rubix\ML\Tests\Persisters;

use Rubix\ML\Encoding;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Persisters\Filesystem;
use PHPUnit\Framework\TestCase;

/**
 * @group Persisters
 * @covers \Rubix\ML\Persisters\Filesystem
 */
class FilesystemTest extends TestCase
{
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

        $this->persister = new Filesystem($this->path, true);
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
        $encoding = new Encoding("Bitch, I'm for real!");

        $this->persister->save($encoding);

        $this->assertFileExists($this->path);

        $encoding = $this->persister->load();

        $this->assertInstanceOf(Encoding::class, $encoding);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFileNotExists($this->path);
    }
}

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
    protected const PATH = __DIR__ . '/test.model';

    /**
     * @var Filesystem
     */
    protected $persister;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->persister = new Filesystem(self::PATH, true);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFileDoesNotExist(self::PATH);
    }

    /**
     * @after
     */
    protected function tearDown() : void
    {
        if (file_exists(self::PATH)) {
            unlink(self::PATH);
        }

        foreach (glob(self::PATH . '*.old') ?: [] as $filename) {
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

        $this->assertFileExists(self::PATH);

        $encoding = $this->persister->load();

        $this->assertInstanceOf(Encoding::class, $encoding);
    }
}

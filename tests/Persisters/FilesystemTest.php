<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Persisters;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Encoding;
use Rubix\ML\Persisters\Filesystem;
use PHPUnit\Framework\TestCase;

#[Group('Persisters')]
#[CoversClass(Filesystem::class)]
class FilesystemTest extends TestCase
{
    protected const string PATH = __DIR__ . '/test.model';

    protected Filesystem $persister;

    protected function setUp() : void
    {
        $this->persister = new Filesystem(path: self::PATH, history: true);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFileDoesNotExist(self::PATH);
    }

    protected function tearDown() : void
    {
        if (file_exists(self::PATH)) {
            unlink(self::PATH);
        }

        foreach (glob(self::PATH . '*.old') ?: [] as $filename) {
            unlink($filename);
        }
    }

    public function testSaveLoad() : void
    {
        $encoding = new Encoding("Bitch, I'm for real!");

        $this->persister->save($encoding);

        $this->assertFileExists(self::PATH);
    }
}

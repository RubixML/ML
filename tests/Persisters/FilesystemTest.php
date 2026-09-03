<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Persisters;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Encoding;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Persisters\Filesystem;
use PHPUnit\Framework\TestCase;

use function file_get_contents;
use function file_put_contents;
use function filesize;
use function glob;
use function str_repeat;
use function sys_get_temp_dir;
use function uniqid;

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

    #[Test]
    public function saveLoad() : void
    {
        $encoding = new Encoding("Bitch, I'm for real!");

        $this->persister->save($encoding);

        $this->assertFileExists(self::PATH);
    }

    #[Test]
    public function saveLargeEncoding() : void
    {
        $encoding = new Encoding(str_repeat('x', 1024 * 1024));

        $this->persister->save($encoding);

        $this->assertFileExists(self::PATH);

        $this->assertSame(1024 * 1024, filesize(self::PATH));
    }

    #[Test]
    public function saveThenLoadRoundTrip() : void
    {
        $data = 'The quick brown fox jumps over the lazy dog.';

        $this->persister->save(new Encoding($data));

        $loaded = $this->persister->load();

        $this->assertSame($data, $loaded->data());
    }

    #[Test]
    public function saveWithHistoryRotatesPrevious() : void
    {
        $this->persister->save(new Encoding('first'));
        $this->persister->save(new Encoding('second'));

        $this->assertSame('second', file_get_contents(self::PATH));

        $old = glob(self::PATH . '-*.old') ?: [];

        $this->assertCount(1, $old);
        $this->assertSame('first', file_get_contents($old[0]));
    }

    #[Test]
    public function constructorRejectsEmptyPath() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Filesystem(path: '');
    }

    #[Test]
    public function constructorRejectsDirectoryPath() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Filesystem(path: sys_get_temp_dir());
    }

    #[Test]
    public function saveEmptyEncoding() : void
    {
        $this->expectException(RuntimeException::class);

        $this->persister->save(new Encoding(''));
    }

    #[Test]
    public function saveToMissingDirectory() : void
    {
        $path = sys_get_temp_dir() . '/rubix-ml-missing-' . uniqid() . '/test.rbx';

        $persister = new Filesystem(path: $path);

        $this->expectException(RuntimeException::class);

        $persister->save(new Encoding('some data'));
    }

    #[Test]
    public function loadMissingFile() : void
    {
        $persister = new Filesystem(path: self::PATH);

        $this->expectException(RuntimeException::class);

        $persister->load();
    }

    #[Test]
    public function loadEmptyFile() : void
    {
        file_put_contents(self::PATH, '');

        $persister = new Filesystem(path: self::PATH);

        $this->expectException(RuntimeException::class);

        $persister->load();
    }
}

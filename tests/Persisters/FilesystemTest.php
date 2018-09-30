<?php

namespace Rubix\Tests\Persisters;

use Rubix\ML\Persisters\Persister;
use Rubix\ML\Persisters\Filesystem;
use PHPUnit\Framework\TestCase;

class FilesystemTest extends TestCase
{
    protected $persister;

    public function setUp()
    {
        $this->persister = new Filesystem('example.temp', true);
    }

    public function test_build_persister()
    {
        $this->assertInstanceOf(Filesystem::class, $this->persister);
        $this->assertInstanceOf(Persister::class, $this->persister);
    }
}

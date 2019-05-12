<?php

namespace Rubix\ML\Tests\Backends;

use Rubix\ML\Backends\Deferred;
use PHPUnit\Framework\TestCase;

class DeferredTest extends TestCase
{
    protected $deferred;

    public function setUp()
    {
        $this->deferred = new Deferred(function ($a, $b) {
            return $a + $b;
        }, [1, 2]);
    }

    public function test_build_deferred()
    {
        $this->assertInstanceOf(Deferred::class, $this->deferred);
    }

    public function test_result()
    {
        $this->assertEquals(3, $this->deferred->result());
    }
}

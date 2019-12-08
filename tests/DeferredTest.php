<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Deferred;
use PHPUnit\Framework\TestCase;

class DeferredTest extends TestCase
{
    /**
     * @var \Rubix\ML\Deferred
     */
    protected $deferred;

    public function setUp() : void
    {
        $this->deferred = new Deferred(function ($a, $b) {
            return $a + $b;
        }, [1, 2]);
    }

    public function test_build_deferred() : void
    {
        $this->assertInstanceOf(Deferred::class, $this->deferred);
        $this->assertIsCallable($this->deferred);
    }

    public function test_compute() : void
    {
        $this->assertEquals(3, $this->deferred->compute());
    }
}

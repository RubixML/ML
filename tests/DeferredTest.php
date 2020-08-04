<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Deferred;
use PHPUnit\Framework\TestCase;

/**
 * @group Other
 * @covers \Rubix\ML\Deferred
 */
class DeferredTest extends TestCase
{
    /**
     * @var \Rubix\ML\Deferred
     */
    protected $deferred;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->deferred = new Deferred(function ($a, $b) {
            return $a + $b;
        }, [1, 2]);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Deferred::class, $this->deferred);
        $this->assertIsCallable($this->deferred);
    }

    /**
     * @test
     */
    public function compute() : void
    {
        $this->assertEquals(3, $this->deferred->compute());
    }
}

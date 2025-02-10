<?php

declare(strict_types=1);

namespace Rubix\ML\Tests;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Deferred;
use PHPUnit\Framework\TestCase;

#[Group('Other')]
#[CoversClass(Deferred::class)]
class DeferredTest extends TestCase
{
    protected Deferred $deferred;

    protected function setUp() : void
    {
        $this->deferred = new Deferred(
            fn: function ($a, $b) {
                return $a + $b;
            },
            args: [1, 2]
        );
    }

    public function testCompute() : void
    {
        $this->assertEquals(3, $this->deferred->compute());
    }
}

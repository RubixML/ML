<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Helpers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Helpers\CPU;
use PHPUnit\Framework\TestCase;

#[Group('Helpers')]
#[CoversClass(CPU::class)]
class CPUTest extends TestCase
{
    #[Test]
    public function cores() : void
    {
        $cores = CPU::cores();

        $this->assertNotNull($cores);
        $this->assertGreaterThan(0, $cores);
    }

    #[Test]
    public function epsilon() : void
    {
        $epsilon = CPU::epsilon();

        $this->assertLessThan(1.0, $epsilon);
        $this->assertGreaterThan(0.0, $epsilon);

        $this->assertFalse(1.0 + $epsilon === 1.0);
    }
}

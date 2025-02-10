<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Strategies;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Strategies\Percentile;
use PHPUnit\Framework\TestCase;

#[Group('Strategies')]
#[CoversClass(Percentile::class)]
class PercentileTest extends TestCase
{
    protected Percentile $strategy;

    protected function setUp() : void
    {
        $this->strategy = new Percentile(50.0);
    }

    public function testFitGuess() : void
    {
        $this->strategy->fit([1, 2, 3, 4, 5]);

        $guess = $this->strategy->guess();

        $this->assertEquals(3, $guess);
    }
}

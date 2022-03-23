<?php

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\TestCase;
use Rubix\ML\Specifications\ProbabilitiesAreNotEmpty;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\ProbabilitiesAreNotEmpty
 */
class ProbabilitiesAreNotEmptyTest extends TestCase
{
    /**
     * @test
     * @dataProvider passesProvider
     *
     * @param \Rubix\ML\Specifications\ProbabilitiesAreNotEmpty $specification
     * @param bool $expected
     */
    public function passes(ProbabilitiesAreNotEmpty $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function passesProvider() : Generator
    {
        yield [
            ProbabilitiesAreNotEmpty::with([0.4, 0.6]),
            true,
        ];

        yield [
            ProbabilitiesAreNotEmpty::with([]),
            false,
        ];
    }
}

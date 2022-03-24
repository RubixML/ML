<?php

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\TestCase;
use Rubix\ML\Specifications\LabelsAreCompatibleWithProbabilities;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\LabelsAreCompatibleWithProbabilities
 */
class LabelsAreCompatibleWithProbabilitiesTest extends TestCase
{
    /**
     * @test
     * @dataProvider passesProvider
     *
     * @param \Rubix\ML\Specifications\LabelsAreCompatibleWithProbabilities $specification
     * @param bool $expected
     */
    public function passes(LabelsAreCompatibleWithProbabilities $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function passesProvider() : Generator
    {
        yield [
            LabelsAreCompatibleWithProbabilities::with(
                [0.4, 0.6],
                ['A', 'B']
            ),
            true,
        ];

        yield [
            LabelsAreCompatibleWithProbabilities::with(
                [0.4, 0.6],
                ['A', 'B', 'C']
            ),
            false,
        ];
    }
}

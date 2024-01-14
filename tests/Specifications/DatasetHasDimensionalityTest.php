<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\DatasetHasDimensionality
 */
class DatasetHasDimensionalityTest extends TestCase
{
    /**
     * @test
     * @dataProvider passesProvider
     *
     * @param DatasetHasDimensionality $specification
     * @param bool $expected
     * @param DatasetHasDimensionality $specification
     */
    public function passes(DatasetHasDimensionality $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function passesProvider() : Generator
    {
        yield [
            DatasetHasDimensionality::with(Unlabeled::quick([
                ['swamp', 'island', 'black knight', 'counter spell'],
            ]), 4),
            true,
        ];

        yield [
            DatasetHasDimensionality::with(Unlabeled::quick([
                [0.0, 1.0, 2.0],
            ]), 4),
            false,
        ];
    }
}

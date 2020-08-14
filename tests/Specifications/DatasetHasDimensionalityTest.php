<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\Specification;
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
     * @param \Rubix\ML\Specifications\Specification $spec
     * @param bool $expected
     */
    public function passes(Specification $spec, bool $expected) : void
    {
        $this->assertSame($expected, $spec->passes());
    }

    /**
     * @return \Generator<array>
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

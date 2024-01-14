<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\DatasetIsNotEmpty
 */
class DatasetIsNotEmptyTest extends TestCase
{
    /**
     * @test
     * @dataProvider passesProvider
     *
     * @param DatasetIsNotEmpty $specification
     * @param bool $expected
     */
    public function passes(DatasetIsNotEmpty $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function passesProvider() : Generator
    {
        yield [
            DatasetIsNotEmpty::with(Unlabeled::quick([
                ['swamp', 'island', 'black knight', 'counter spell'],
            ])),
            true,
        ];

        yield [
            DatasetIsNotEmpty::with(Unlabeled::quick()),
            false,
        ];
    }
}

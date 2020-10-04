<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\DatasetIsLabeled;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\DatasetIsLabeled
 */
class DatasetIsLabeledTest extends TestCase
{
    /**
     * @test
     * @dataProvider passesProvider
     *
     * @param \Rubix\ML\Specifications\DatasetIsLabeled $specification
     * @param bool $expected
     */
    public function passes(DatasetIsLabeled $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }

    /**
     * @return \Generator<array>
     */
    public function passesProvider() : Generator
    {
        yield [
            DatasetIsLabeled::with(Labeled::quick([
                ['swamp', 'island', 'black knight', 'counter spell'],
            ], ['win'])),
            true,
        ];

        yield [
            DatasetIsLabeled::with(Unlabeled::quick([
                ['swamp', 'island', 'black knight', 'counter spell'],
            ])),
            false,
        ];
    }
}

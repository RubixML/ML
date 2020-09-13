<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\Specification;
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

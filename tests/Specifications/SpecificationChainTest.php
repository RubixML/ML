<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\DatasetIsNotEmpty
 */
class SpecificationChainTest extends TestCase
{
    /**
     * @test
     * @dataProvider passesProvider
     *
     * @param \Rubix\ML\Specifications\SpecificationChain $specification
     * @param bool $expected
     */
    public function passes(SpecificationChain $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }

    /**
     * @return \Generator<array>
     */
    public function passesProvider() : Generator
    {
        $dataset = Unlabeled::quick([
            ['swamp', 'island', 'black knight', 'counter spell'],
        ]);

        yield [
            SpecificationChain::with([
                new DatasetIsNotEmpty($dataset),
            ]),
            true,
        ];

        yield [
            SpecificationChain::with([
                new DatasetIsNotEmpty($dataset),
                new DatasetIsLabeled($dataset),
            ]),
            false,
        ];
    }
}

<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\Specification;
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

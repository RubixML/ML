<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[CoversClass(DatasetIsNotEmpty::class)]
class SpecificationChainTest extends TestCase
{
    public static function passesProvider() : Generator
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

    /**
     * @param SpecificationChain $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    public function testPasses(SpecificationChain $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}

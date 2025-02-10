<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[CoversClass(DatasetHasDimensionality::class)]
class DatasetHasDimensionalityTest extends TestCase
{
    public static function passesProvider() : Generator
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

    /**
     * @param DatasetHasDimensionality $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    public function testPasses(DatasetHasDimensionality $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}

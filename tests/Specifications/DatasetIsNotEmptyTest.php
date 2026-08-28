<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[CoversClass(DatasetIsNotEmpty::class)]
class DatasetIsNotEmptyTest extends TestCase
{
    public static function passesProvider() : Generator
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

    /**
     * @param DatasetIsNotEmpty $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    public function testPasses(DatasetIsNotEmpty $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}

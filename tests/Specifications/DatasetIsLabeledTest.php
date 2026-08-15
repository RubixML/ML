<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\DatasetIsLabeled;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[CoversClass(DatasetIsLabeled::class)]
class DatasetIsLabeledTest extends TestCase
{
    public static function passesProvider() : Generator
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

    /**
     * @param DatasetIsLabeled $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    public function testPasses(DatasetIsLabeled $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}

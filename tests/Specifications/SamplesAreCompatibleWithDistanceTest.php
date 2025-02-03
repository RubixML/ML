<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Kernels\Distance\Hamming;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Specifications\SamplesAreCompatibleWithDistance;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[CoversClass(SamplesAreCompatibleWithDistance::class)]
class SamplesAreCompatibleWithDistanceTest extends TestCase
{
    public static function passesProvider() : Generator
    {
        yield [
            SamplesAreCompatibleWithDistance::with(
                Unlabeled::quick([
                    ['swamp', 'island', 'black knight', 'counter spell'],
                ]),
                new Hamming()
            ),
            true,
        ];

        yield [
            SamplesAreCompatibleWithDistance::with(
                Unlabeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ]),
                new Euclidean()
            ),
            false,
        ];

        yield [
            SamplesAreCompatibleWithDistance::with(
                Unlabeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ]),
                new Hamming()
            ),
            false,
        ];

        yield [
            SamplesAreCompatibleWithDistance::with(
                Unlabeled::quick([
                    [1, 2, 3, 4, 5],
                ]),
                new Euclidean()
            ),
            true,
        ];
    }

    /**
     * @param SamplesAreCompatibleWithDistance $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    public function passes(SamplesAreCompatibleWithDistance $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}

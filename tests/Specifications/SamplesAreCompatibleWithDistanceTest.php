<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Kernels\Distance\Hamming;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Specifications\Specification;
use Rubix\ML\Specifications\SamplesAreCompatibleWithDistance;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\SamplesAreCompatibleWithDistance
 */
class SamplesAreCompatibleWithDistanceTest extends TestCase
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
}

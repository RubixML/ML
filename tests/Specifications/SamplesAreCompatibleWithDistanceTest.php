<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Kernels\Distance\Hamming;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Specifications\SamplesAreCompatibleWithDistance;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\SamplesAreCompatibleWithDistance
 */
class SamplesAreCompatibleWithDistanceTest extends TestCase
{
    /**
     * @test
     * @dataProvider checkProvider
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @param bool $valid
     */
    public function check(Dataset $dataset, Distance $kernel, bool $valid) : void
    {
        if (!$valid) {
            $this->expectException(InvalidArgumentException::class);
        }

        SamplesAreCompatibleWithDistance::check($dataset, $kernel);

        $this->assertTrue($valid);
    }

    /**
     * @return \Generator<array>
     */
    public function checkProvider() : Generator
    {
        yield [
            Unlabeled::quick([
                ['swamp', 'island', 'black knight', 'counter spell'],
            ]),
            new Hamming(),
            true,
        ];

        yield [
            Unlabeled::quick([
                [6.0, -1.1, 5, 'college'],
            ]),
            new Euclidean(),
            false,
        ];

        yield [
            Unlabeled::quick([
                [6.0, -1.1, 5, 'college'],
            ]),
            new Hamming(),
            false,
        ];

        yield [
            Unlabeled::quick([
                [1, 2, 3, 4, 5],
            ]),
            new Euclidean(),
            true,
        ];
    }
}

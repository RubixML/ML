<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Specifications\SamplesAreCompatibleWithDistance;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\SamplesAreCompatibleWithDistance
 */
class SamplesAreCompatibleWithDistanceTest extends TestCase
{
    /**
     * @test
     */
    public function check() : void
    {
        $kernel = new Euclidean();

        $dataset = Unlabeled::quick([
            [6., -1.1, 5, 'college'],
        ]);

        $this->expectException(InvalidArgumentException::class);

        SamplesAreCompatibleWithDistance::check($dataset, $kernel);
    }
}

<?php

namespace Rubix\ML\Tests\Other\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithDistance;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class SamplesAreCompatibleWithDistanceTest extends TestCase
{
    public function test_check() : void
    {
        $kernel = new Euclidean();

        $dataset = Unlabeled::quick([
            [6., -1.1, 5, 'college'],
        ]);

        $this->expectException(InvalidArgumentException::class);

        SamplesAreCompatibleWithDistance::check($dataset, $kernel);
    }
}

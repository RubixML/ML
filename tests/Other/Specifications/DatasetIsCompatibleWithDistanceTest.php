<?php

namespace Rubix\ML\Tests\Other\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithDistance;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class DatasetIsCompatibleWithDistanceTest extends TestCase
{
    public function test_check()
    {
        $kernel = new Euclidean();

        $dataset = Unlabeled::quick([
            [6., -1.1, 5, 'college'],
        ]);

        $this->expectException(InvalidArgumentException::class);

        DatasetIsCompatibleWithDistance::check($dataset, $kernel);
    }
}

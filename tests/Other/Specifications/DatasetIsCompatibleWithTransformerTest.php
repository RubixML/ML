<?php

namespace Rubix\ML\Tests\Other\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\L1Normalizer;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithTransformer;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class DatasetIsCompatibleWithTransformerTest extends TestCase
{
    public function test_check() : void
    {
        $transformer = new L1Normalizer();

        $dataset = Unlabeled::quick([
            [6., -1.1, 5, 'college'],
        ]);

        $this->expectException(InvalidArgumentException::class);

        DatasetIsCompatibleWithTransformer::check($dataset, $transformer);
    }
}

<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\L1Normalizer;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer
 */
class SamplesAreCompatibleWithTransformerTest extends TestCase
{
    /**
     * @test
     */
    public function check() : void
    {
        $transformer = new L1Normalizer();

        $dataset = Unlabeled::quick([
            [6., -1.1, 5, 'college'],
        ]);

        $this->expectException(InvalidArgumentException::class);

        SamplesAreCompatibleWithTransformer::check($dataset, $transformer);
    }
}

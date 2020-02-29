<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Embedders\TSNE;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEmbedder;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\SamplesAreCompatibleWithEmbedder
 */
class SamplesAreCompatibleWithEmbedderTest extends TestCase
{
    /**
     * @test
     */
    public function check() : void
    {
        $embedder = new TSNE();

        $dataset = Unlabeled::quick([
            [6., -1.1, 5, 'college'],
        ]);

        $this->expectException(InvalidArgumentException::class);

        SamplesAreCompatibleWithEmbedder::check($dataset, $embedder);
    }
}

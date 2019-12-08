<?php

namespace Rubix\ML\Tests\Other\Specifications;

use Rubix\ML\Embedders\TSNE;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEmbedder;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class DatasetIsCompatibleWithEmbedderTest extends TestCase
{
    public function test_check() : void
    {
        $embedder = new TSNE();

        $dataset = Unlabeled::quick([
            [6., -1.1, 5, 'college'],
        ]);

        $this->expectException(InvalidArgumentException::class);

        DatasetIsCompatibleWithEmbedder::check($dataset, $embedder);
    }
}

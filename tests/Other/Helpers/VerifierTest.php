<?php

namespace Rubix\ML\Tests\Other\Helpers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Helpers\Verifier;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use PHPUnit\Framework\TestCase;
use Exception;

/**
 * @group Helpers
 * @covers \Rubix\ML\Other\Helpers\Verifier
 */
class VerifierTest extends TestCase
{
    /**
     * @test
     */
    public function check() : void
    {
        $dataset = Unlabeled::quick([
            ['light pillar', 'desert spirit'],
        ]);

        $this->expectException(Exception::class);

        Verifier::check([
            DatasetIsNotEmpty::with($dataset),
            DatasetHasDimensionality::with($dataset, 3),
        ]);
    }
}

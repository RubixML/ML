<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\L1Normalizer;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(L1Normalizer::class)]
class L1NormalizerTest extends TestCase
{
    protected L1Normalizer $transformer;

    protected function setUp() : void
    {
        $this->transformer = new L1Normalizer();
    }

    public function testTransform() : void
    {
        $dataset = new Unlabeled(samples: [
            [1, 2, 3, 4],
            [40, 0, 30, 10],
            [100, 300, 200, 400],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.0, 0.375, 0.125],
            [0.1, 0.3, 0.2, 0.4],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}

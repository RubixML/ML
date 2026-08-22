<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\L2Normalizer;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(L2Normalizer::class)]
class L2NormalizerTest extends TestCase
{
    protected L2Normalizer $transformer;

    protected function setUp() : void
    {
        $this->transformer = new L2Normalizer();
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
            [0.18257418583505536, 0.3651483716701107, 0.5477225575051661, 0.7302967433402214],
            [0.7844645405527362, 0.0, 0.5883484054145521, 0.19611613513818404],
            [0.18257418583505536, 0.5477225575051661, 0.3651483716701107, 0.7302967433402214],
        ];

        $this->assertEqualsWithDelta($expected, $dataset->samples(), 1e-8);
    }
}

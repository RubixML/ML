<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\PolynomialExpander;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(PolynomialExpander::class)]
class PolynomialExpanderTest extends TestCase
{
    protected PolynomialExpander $transformer;

    protected function setUp() : void
    {
        $this->transformer = new PolynomialExpander(2);
    }

    public function testTransform() : void
    {
        $dataset = new Unlabeled(samples: [
            [1, 2, 3, 4],
            [40, 20, 30, 10],
            [100, 300, 200, 400],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            [1, 1, 2, 4, 3, 9, 4, 16],
            [40, 1600, 20, 400, 30, 900, 10, 100],
            [100, 10000, 300, 90000, 200, 40000, 400, 160000],
        ];

        $this->assertEqualsWithDelta($expected, $dataset->samples(), 1e-8);
    }
}

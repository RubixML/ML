<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Placeholder1D::class)]
class Placeholder1DTest extends TestCase
{
    protected Matrix $input;

    protected Placeholder1D $layer;

    protected function setUp() : void
    {
        $this->input = Matrix::quick([
            [1.0, 2.5],
            [0.1, 0.0],
            [0.002, -6.0],
        ]);

        $this->layer = new Placeholder1D(3);
    }

    public function testForwardInfer() : void
    {
        $this->assertEquals(3, $this->layer->width());

        $expected = [
            [1.0, 2.5],
            [0.1, 0.0],
            [0.002, -6.0],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertEquals($expected, $forward->asArray());

        $infer = $this->layer->infer($this->input);

        $this->assertEquals($expected, $infer->asArray());
    }
}

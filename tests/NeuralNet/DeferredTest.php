<?php

namespace Rubix\ML\Tests\NeuralNet;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Deferred;
use PHPUnit\Framework\TestCase;

class DeferredTest extends TestCase
{
    protected $deferred;

    public function setUp()
    {
        $this->deferred = new Deferred(function () {
            return new Matrix([
                [-2, 5],
                [-7, 9],
            ]);
        });
    }

    public function test_build_deferred()
    {
        $this->assertInstanceOf(Deferred::class, $this->deferred);
    }

    public function test_result()
    {
        $expected = [
            [-2, 5],
            [-7, 9],
        ];

        $this->assertInstanceOf(Matrix::class, $this->deferred->result());
        $this->assertEquals($expected, $this->deferred->result()->asArray());
    }
}

<?php

namespace Rubix\ML\Tests\Kernels\SVM;

use Rubix\ML\Kernels\SVM\Polynomial;
use Rubix\ML\Kernels\SVM\Kernel;
use PHPUnit\Framework\TestCase;

/**
 * @group Kernels
 * @requires extension svm
 * @covers \Rubix\ML\Kernels\SVM\Polynomial
 */
class PolynomialTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\SVM\Polynomial
     */
    protected $kernel;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->kernel = new Polynomial(3, 1e-3);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Polynomial::class, $this->kernel);
        $this->assertInstanceOf(Kernel::class, $this->kernel);
    }

    /**
     * @test
     */
    public function options() : void
    {
        $expected = [
            102 => 1,
            201 => 1e-3,
            103 => 3,
            205 => 0.0,
        ];

        $this->assertEquals($expected, $this->kernel->options());
    }
}

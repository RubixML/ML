<?php

namespace Rubix\ML\Tests\Kernels\SVM;

use Rubix\ML\Kernels\SVM\Linear;
use Rubix\ML\Kernels\SVM\Kernel;
use PHPUnit\Framework\TestCase;

/**
 * @group Kernels
 * @requires extension svm
 * @covers \Rubix\ML\Kernels\SVM\Linear
 */
class LinearTest extends TestCase
{
    /**
     * @var Linear
     */
    protected $kernel;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->kernel = new Linear();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Linear::class, $this->kernel);
        $this->assertInstanceOf(Kernel::class, $this->kernel);
    }

    /**
     * @test
     */
    public function options() : void
    {
        $expected = [102 => 0];

        $this->assertEquals($expected, $this->kernel->options());
    }
}

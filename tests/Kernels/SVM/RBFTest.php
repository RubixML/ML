<?php

namespace Rubix\ML\Tests\Kernels\SVM;

use Rubix\ML\Kernels\SVM\RBF;
use Rubix\ML\Kernels\SVM\Kernel;
use PHPUnit\Framework\TestCase;

/**
 * @group Kernels
 * @requires extension svm
 * @covers \Rubix\ML\Kernels\SVM\RBF
 */
class RBFTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\SVM\RBF
     */
    protected $kernel;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->kernel = new RBF(1e-3);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(RBF::class, $this->kernel);
        $this->assertInstanceOf(Kernel::class, $this->kernel);
    }

    /**
     * @test
     */
    public function options() : void
    {
        $options = [
            102 => 2,
            201 => 1e-3,
        ];

        $this->assertEquals($options, $this->kernel->options());
    }
}

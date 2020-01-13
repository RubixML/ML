<?php

namespace Rubix\ML\Tests\Kernels\SVM;

use Rubix\ML\Kernels\SVM\Sigmoidal;
use Rubix\ML\Kernels\SVM\Kernel;
use PHPUnit\Framework\TestCase;

/**
 * @group Kernels
 * @requires extension svm
 * @covers \Rubix\ML\Kernels\SVM\Sigmoidal
 */
class SigmoidalTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\SVM\Sigmoidal
     */
    protected $kernel;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->kernel = new Sigmoidal(1e-3);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Sigmoidal::class, $this->kernel);
        $this->assertInstanceOf(Kernel::class, $this->kernel);
    }

    /**
     * @test
     */
    public function options() : void
    {
        $options = [
            102 => 3,
            201 => 1e-3,
            205 => 0.0,
        ];

        $this->assertEquals($options, $this->kernel->options());
    }
}

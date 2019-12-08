<?php

namespace Rubix\ML\Tests\Kernels\SVM;

use Rubix\ML\Kernels\SVM\Sigmoidal;
use Rubix\ML\Kernels\SVM\Kernel;
use PHPUnit\Framework\TestCase;

class SigmoidalTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\SVM\Sigmoidal
     */
    protected $kernel;

    public function setUp() : void
    {
        $this->kernel = new Sigmoidal(1e-3);
    }

    public function test_build_svm_kernel() : void
    {
        $this->assertInstanceOf(Sigmoidal::class, $this->kernel);
        $this->assertInstanceOf(Kernel::class, $this->kernel);
    }

    public function test_get_options() : void
    {
        $options = [
            102 => 3,
            201 => 1e-3,
            205 => 0.,
        ];

        $this->assertEquals($options, $this->kernel->options());
    }
}

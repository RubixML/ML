<?php

namespace Rubix\ML\Tests\Kernels\SVM;

use Rubix\ML\Kernels\SVM\RBF;
use Rubix\ML\Kernels\SVM\Kernel;
use PHPUnit\Framework\TestCase;

class RBFTest extends TestCase
{
    protected $kernel;

    public function setUp()
    {
        $this->kernel = new RBF(1e-3);
    }

    public function test_build_svm_kernel()
    {
        $this->assertInstanceOf(RBF::class, $this->kernel);
        $this->assertInstanceOf(Kernel::class, $this->kernel);
    }

    public function test_get_options()
    {
        $options = [
            102 => 2,
            201 => 1e-3,
        ];

        $this->assertEquals($options, $this->kernel->options());
    }
}

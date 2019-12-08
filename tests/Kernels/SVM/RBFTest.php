<?php

namespace Rubix\ML\Tests\Kernels\SVM;

use Rubix\ML\Kernels\SVM\RBF;
use Rubix\ML\Kernels\SVM\Kernel;
use PHPUnit\Framework\TestCase;

class RBFTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\SVM\RBF
     */
    protected $kernel;

    public function setUp() : void
    {
        $this->kernel = new RBF(1e-3);
    }

    public function test_build_svm_kernel() : void
    {
        $this->assertInstanceOf(RBF::class, $this->kernel);
        $this->assertInstanceOf(Kernel::class, $this->kernel);
    }

    public function test_get_options() : void
    {
        $options = [
            102 => 2,
            201 => 1e-3,
        ];

        $this->assertEquals($options, $this->kernel->options());
    }
}

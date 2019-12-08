<?php

namespace Rubix\ML\Tests\Kernels\SVM;

use Rubix\ML\Kernels\SVM\Linear;
use Rubix\ML\Kernels\SVM\Kernel;
use PHPUnit\Framework\TestCase;

class LinearTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\SVM\Linear
     */
    protected $kernel;

    public function setUp() : void
    {
        $this->kernel = new Linear();
    }

    public function test_build_svm_kernel() : void
    {
        $this->assertInstanceOf(Linear::class, $this->kernel);
        $this->assertInstanceOf(Kernel::class, $this->kernel);
    }

    public function test_get_options() : void
    {
        $options = [102 => 0];

        $this->assertEquals($options, $this->kernel->options());
    }
}

<?php

namespace Rubix\ML\Tests\Kernels\SVM;

use Rubix\ML\Kernels\SVM\Polynomial;
use Rubix\ML\Kernels\SVM\Kernel;
use PHPUnit\Framework\TestCase;

class PolynomialTest extends TestCase
{
    protected $kernel;

    public function setUp()
    {
        $this->kernel = new Polynomial(3, 1e-3);
    }

    public function test_build_svm_kernel()
    {
        $this->assertInstanceOf(Polynomial::class, $this->kernel);
        $this->assertInstanceOf(Kernel::class, $this->kernel);
    }

    public function test_get_options()
    {
        $options = [
            102 => 1,
            201 => 1e-3,
            103 => 3,
            205 => 0.,
        ];

        $this->assertEquals($options, $this->kernel->options());
    }
}

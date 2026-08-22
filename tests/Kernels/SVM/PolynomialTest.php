<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Kernels\SVM;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Kernels\SVM\Polynomial;
use PHPUnit\Framework\TestCase;

#[Group('Kernels')]
#[RequiresPhpExtension('svm')]
#[CoversClass(Polynomial::class)]
class PolynomialTest extends TestCase
{
    protected Polynomial $kernel;

    protected function setUp() : void
    {
        $this->kernel = new Polynomial(degree: 3, gamma: 1e-3);
    }

    public function testOptions() : void
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

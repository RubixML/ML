<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Kernels\SVM;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Kernels\SVM\Linear;
use PHPUnit\Framework\TestCase;

#[Group('Kernels')]
#[RequiresPhpExtension('svm')]
#[CoversClass(Linear::class)]
class LinearTest extends TestCase
{
    protected Linear $kernel;

    protected function setUp() : void
    {
        $this->kernel = new Linear();
    }

    public function testOptions() : void
    {
        $expected = [102 => 0];

        $this->assertEquals($expected, $this->kernel->options());
    }
}

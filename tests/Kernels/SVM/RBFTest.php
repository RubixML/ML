<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Kernels\SVM;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Kernels\SVM\RBF;
use PHPUnit\Framework\TestCase;

#[Group('Kernels')]
#[RequiresPhpExtension('svm')]
#[CoversClass(RBF::class)]
class RBFTest extends TestCase
{
    protected RBF $kernel;

    protected function setUp() : void
    {
        $this->kernel = new RBF(1e-3);
    }

    public function testOptions() : void
    {
        $options = [
            102 => 2,
            201 => 1e-3,
        ];

        $this->assertEquals($options, $this->kernel->options());
    }
}
